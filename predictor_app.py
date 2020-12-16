import streamlit as st
import lazypredict
import pandas as pd
#from io import StringIO
from sklearn.model_selection import train_test_split
st.title("auto-predictor application")
st.write("Lazypredict module is used here")
st.write("visit here to Learn about lazypredict: https://github.com/shankarpandala/lazypredict")
action = st.radio("Choose your action:",options = ['classification',
                                                  'regression'])
if action == 'classification':
    st.write('classification task has been chosen')
elif action == 'regression':
    st.write('regression task has been chosen')
else:
    pass
st.write("Upload your data file below:")
st.write("caution: we can't process more than 200 MB")
uploaded_file = st.file_uploader(label = "choose your data file")
if uploaded_file is not None:
    #to read file as bytes:
    dataframe = pd.read_csv(uploaded_file)
    st.write("thanks for uploading the file. Processing...")

    st.subheader("the data looks like this:")
    st.write(dataframe.head())
    st.subheader("Name the target column")
    target_column = st.text_input("Target column name","Example: Target")
    Y = dataframe[target_column]
    X = dataframe.drop(target_column,axis = 1)
    st.write("choose the train_test_split value")
    ratio = st.slider("How much test percentage to keep:",0.01,0.5,0.01)
    st.write("ratio chosen is:",ratio)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = ratio, random_state = 42)

    if action == 'classification':
        from lazypredict.Supervised import LazyClassifier
        clf = LazyClassifier(verbose = 0, ignore_warnings = True,custom_metric = None)
        models,predictions = clf.fit(X_train,X_test,Y_train,Y_test)
        st.write(models)
    elif action == 'regression':
        from lazypredict.Supervised import LazyRegressor
        reg = LazyRegressor(verbose = 0, ifnore_warnings = False, custom_metric = None)
        models,predictions = reg.fit(X_train,X_test,Y_train,Y_test)
        st.write(models)
    st.write("Now, you can choose the best model and go on modeling on your own")
    st.write("thanks for visiting")
else:
    st.write("You do have to upload a file to proceed")
