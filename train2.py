from sklearn.kernel_ridge import KernelRidge
from misc import load_data, split_features_target, split_train_test, build_pipeline, train_pipeline, evaluate_pipeline, cross_val_mse

def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = KernelRidge(alpha=1.0, kernel='rbf')
    pipeline = build_pipeline(model)
    pipeline = train_pipeline(pipeline, X_train, y_train)
    mse_test = evaluate_pipeline(pipeline, X_test, y_test)
    cv_mse = cross_val_mse(pipeline, X, y)
    print("KernelRidge - Test MSE: {:.4f}".format(mse_test))
    print("KernelRidge - 5-fold CV mean MSE: {:.4f}".format(cv_mse))

if __name__ == "__main__":
    main()
