from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_features_target, split_train_test, build_pipeline, train_pipeline, evaluate_pipeline, cross_val_mse

def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = DecisionTreeRegressor(random_state=42)
    pipeline = build_pipeline(model)
    pipeline = train_pipeline(pipeline, X_train, y_train)
    mse_test = evaluate_pipeline(pipeline, X_test, y_test)
    cv_mse = cross_val_mse(pipeline, X, y)
    print("DecisionTreeRegressor - Test MSE: {:.4f}".format(mse_test))
    print("DecisionTreeRegressor - 5-fold CV mean MSE: {:.4f}".format(cv_mse))

if __name__ == "__main__":
    main()
