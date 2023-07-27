import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
import seaborn as sns
import re
import warnings
from matplotlib.gridspec import GridSpec
from pprint import pprint
import scipy
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import squarify
import time
import joblib
import streamlit as st
import pickle
import PIL

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix, classification_report
from sklearn.utils import resample


data = pd.read_csv(filepath_or_buffer='bank-marketing.csv', sep=';')
df = data.copy()

pd.set_option('display.max_columns', None)
columns_names = [re.sub('\.', '_', column.lower()) for column in df.columns]
df.columns = columns_names

df['job'] = df['job'].str.replace('admin.', 'admin')
df['education'] = df['education'].str.replace('.', '_')
df['poutcome'] = df['poutcome'].str.replace('nonexistent', 'unknown')

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor()
lof.fit_predict(df.select_dtypes(exclude=['object']))
df_scores = lof.negative_outlier_factor_

# scores = pd.DataFrame(np.sort(df_scores))
# scores.plot(stacked=True, xlim=[0, 50], style='.-')
# plt.show()
threshold = np.sort(df_scores)[15]
df.drop(index=df[df_scores < threshold].index, inplace=True)
df.reset_index(inplace=True, drop=True)

df['pdays'] = np.where(df['pdays'] > 500, 1, 0)
df['contact'] = np.where(df['contact'] == 'cellular', 1, 0)
df['y'] = np.where(df['y'] == 'yes', 1, 0)
df = pd.get_dummies(df, drop_first=True, dtype='int')

df_majority = df[df['y'] == 0]
df_minority = df[df['y'] == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=36548,
                                 random_state=42)

df_upsampled = pd.concat([df_minority_upsampled, df_majority])

def standardization_split(dataframe, dependent, test_size):
    scaler = StandardScaler()

    y = dataframe[[dependent]]
    X = dataframe.drop([dependent], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler

X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler_train = standardization_split(df_upsampled, 'y', 0.2)

def decomposition_df(train, test, n_components):
    pca = PCA(n_components=n_components)

    pca.fit(train)

    X_pca_train = pca.transform(train)
    X_pca_test = pca.transform(test)

    explained_variance_ratio = pca.explained_variance_ratio_
    return X_pca_train, X_pca_test, explained_variance_ratio, pca

X_pca_train, X_pca_test, \
    explained_variance_ratio, pca_train = decomposition_df(X_train_scaled, X_test_scaled, 20)

# model = KNeighborsClassifier().fit(X_pca_train, y_train)
# print(accuracy_score(y_test, model.predict(X_pca_test)))
# print("*" * 50)
# print(classification_report(y_test, model.predict(X_pca_test)))
# print("*" * 50)
# print(confusion_matrix(y_test, model.predict(X_pca_test)))

#Save the model
# filename = 'bank_model.pkl'
# joblib.dump(model, filename)

#Load the model
model = joblib.load('bank_model.pkl')

##
bank_image = PIL.Image.open(fp = 'features_marketing_banks_blog_948979b181.png')

interface = st.container()
with interface:
    st.title(body='Bank Marketing') #Page name

    st.image(image=bank_image)

    st.header(body='Project Description') #Caption

    st.markdown(body=f"""This is a machine learning project ...""")

    st.subheader(body='Input Features')

    st.markdown(body = '***')

    age = st.slider(label='Age of the customer:', min_value=0, max_value=100, value=int(df.age.mean()))

    categorical_options_job = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    #selected_option_job = st.radio('The type of job the customer has:', categorical_options_job)
    selected_option_job = st.selectbox('The type of job the customer has:', categorical_options_job)

    categorical_options_marital = ['divorced', 'married', 'single', 'unknown']
    selected_option_marital = st.selectbox('The marital status of the customer:', categorical_options_marital)

    categorical_options_education = ['basic_4y', 'basic_6y', 'basic_9y', 'high_school', 'illiterate', 'professional_course', 'university_degree', 'unknown']
    selected_option_education = st.selectbox('The level of education of the customer:', categorical_options_education)

    categorical_options_default = ['no', 'yes', 'unknown']
    selected_option_default = st.selectbox('The customer has credit in default:', categorical_options_default)

    categorical_options_housing = ['no', 'yes', 'unknown']
    selected_option_housing = st.selectbox('The customer has a housing loan:', categorical_options_housing)

    categorical_options_loan = ['no', 'yes', 'unknown']
    selected_option_loan = st.selectbox('The customer has a personal loan:', categorical_options_loan)

    categorical_options_contact = ['cellular', 'telephone']
    selected_option_contact = st.selectbox('The contact communication type:', categorical_options_contact)

    categorical_options_month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    selected_option_month = st.selectbox('The month of the year when the customer was last contacted:', categorical_options_month)

    categorical_options_day_of_week = ['mon', 'tue', 'wed', 'thu', 'fri']
    selected_option_day_of_week = st.selectbox('Last contact day of the week:', categorical_options_day_of_week)

    duration = st.slider(label='The duration of the last contact in seconds:', min_value=0, max_value=5000, value=int(df.duration.mean()))

    pdays = st.slider(label='The number of days that passed by after the customer was last contacted from a previous campaign:', min_value=-1, max_value=999, value=int(df.pdays.mean()))

    previous = st.slider(label='The number of contacts performed before this campaign and for this customer:', min_value=0, max_value=7, value=int(df.previous.mean()))

    categorical_options_poutcome = ['failure', 'unknown', 'success']
    selected_option_poutcome = st.selectbox('The outcome of the previous marketing campaign:', categorical_options_poutcome)

    emp_var_rate = st.slider(label='Employment variation rate - quarterly indicator:', min_value=-4.0, max_value=2.0, value=df['emp_var_rate'].mean(), step=0.1)

    cons_price_idx = st.slider(label='Consumer price index - monthly indicator:', min_value=91.0, max_value=95.0, value=df['cons_price_idx'].mean(), step=0.00001)

    cons_conf_idx = st.slider(label='Consumer confidence index - monthly indicator:', min_value=-55.0, max_value=-20.0, value=df['cons_conf_idx'].mean(), step=0.1)

    euribor3m = st.slider(label='Euribor 3 months rate - daily indicator:', min_value=0.0, max_value=6.0, value=df.euribor3m.mean(), step=00.1)

    nr_employed = st.slider(label='Number of employees - quarterly indicator:', min_value=1000, max_value=10000, value=int(df['nr_employed'].mean()))


    st.markdown(body = '***')

    st.subheader(body='Making Predictions')

    data_dicionary = {
                    'age': age,
                    'job': selected_option_job,
                    'marital': selected_option_marital,
                    'education': selected_option_education,
                    'default': selected_option_default,
                    'housing': selected_option_housing,
                    'loan': selected_option_loan,
                    'contact': selected_option_contact,
                    'month': selected_option_month,
                    'day_of_week': selected_option_day_of_week,
                    'duration': duration,
                    'pdays': pdays,
                    'previous': previous,
                    'poutcome': selected_option_poutcome,
                    'emp_var_rate': emp_var_rate,
                    'cons_price_idx': cons_price_idx,
                    'cons_conf_idx': cons_conf_idx,
                    'euribor3m': euribor3m,
                    'nr_employed': nr_employed}

    input_features = pd.DataFrame(data=data_dicionary, index=[0])

    # model = joblib.load('bank_model.pkl')

    input_features['pdays'] = np.where(input_features['pdays'] == 999, 1, 0)
    input_features['contact'] = np.where(input_features['contact'] == 'cellular', 1, 0)

    input_features = pd.get_dummies(input_features, drop_first=True)

    training_columns = set(X_train.columns)

    input_features = input_features.reindex(columns=X_train.columns, fill_value=0)
    input_features_scaled = scaler_train.transform(input_features)

    input_features_pca = pca_train.transform(input_features_scaled)

    #Make Prediction
    if st.button('Predict'):
        with st.spinner(text='Sending input features to model...'):
            time.sleep(2)

        predicted_value = model.predict(input_features_pca)

        st.success('Prediction is ready')

        time.sleep(1)
        if predicted_value[0] == 1:
            st.markdown("<h4>Model output: The customer will subscribe to a term deposit</h4>", unsafe_allow_html=True)

        else:
            st.markdown("<h4>Model output: The customer will **not** subscribe to a term deposit</h4>", unsafe_allow_html=True)

