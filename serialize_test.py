# filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # # Load up the model and try a new test sample with it
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    # #result.predict(X_test)
    # print(result)

    # Serialize using joblib
    joblib.dump(model, 'finalized_model.pkl')
    model_from_joblib = joblib.load('finalized_model.pkl')
    model_from_joblib.predict(x_test)