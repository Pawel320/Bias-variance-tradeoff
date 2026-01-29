import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

rnd = np.random.RandomState(42)

# h(x) function
def h_of_x1(x1):
    return np.sin(2 * x1) + x1 * np.cos(x1 - 1)


class SklearnWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):

        if isinstance(self.model, KNeighborsRegressor):
            self.model.n_neighbors = min(self.model.n_neighbors, len(X))

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def tune_parameter(model_name, candidate_values, N=80, p=5, sigma=1.0):
    X_train = rnd.uniform(-10, 10, size=(N, p))
    y_train = h_of_x1(X_train[:, 0]) + rnd.normal(0, sigma, size=N)

    X_val = rnd.uniform(-10, 10, size=(N, p))
    y_val = h_of_x1(X_val[:, 0]) 

    best_param = None
    best_error = float("inf")

    for val in candidate_values:

        if model_name == "ridge":
            model = Ridge(alpha=val)
        elif model_name == "knn":
            k = min(val, N) # k must be smaller than the set size
            model = KNeighborsRegressor(n_neighbors=k)

        elif model_name == "tree":
            model = DecisionTreeRegressor(max_depth=val, random_state=0)
        else:
            raise ValueError("unknown model")

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = np.mean((preds - y_val)**2)

        if mse < best_error:
            best_error = mse
            best_param = val

    return best_param

# best (tuned) parameters
BEST_DEFAULTS = {
    "ridge": tune_parameter("ridge", [0.01, 0.1, 0.5, 1, 5, 10,15]),
    "knn":   tune_parameter("knn", [1,2,3,4,5,7,9,11,13,15,19,21,23,30,40]),
    "tree":  tune_parameter("tree", [2,4,6,8,10,12,15,20,25,30])
}


# takes model name and suggested parameters, returns full model
def get_model(model_name, **kwargs):
    if model_name == "ridge":
        alpha = kwargs.get("alpha", BEST_DEFAULTS["ridge"])
        return SklearnWrapper(Ridge(alpha=alpha))
    
    if model_name == "knn":
        k = kwargs.get("k", BEST_DEFAULTS["knn"])
        return SklearnWrapper(KNeighborsRegressor(n_neighbors=k))
    
    if model_name == "tree":
        depth = kwargs.get("max_depth", BEST_DEFAULTS["tree"])
        return SklearnWrapper(DecisionTreeRegressor(max_depth=depth, random_state=0))

    raise ValueError("Unknown model")


# ------------------
# -----TASK 2.1-----
# ------------------
# Protocol for estimating squared bias, variance, and residual error at point x:
# 1. We generate M independent training sets (LS) with N samples each
# 2. For each set, we train a selected regression model
# 3. We predict the value at point x for each previously trained model
# 4. The average of these predictions gives an estimate of the expected prediction
# 5. We calculate statistics such as squared bias, variance,8 residual and expected error.

def compute_bias_variance_at_x(model_name, point, M=60, N=80, p=5, sigma=1.0):

    point = np.array(point).reshape(1, -1)

    predictions = []
    true_h = h_of_x1(point[0, 0])

    # calculate mean error over M models
    for m in range(M):
        X_train = rnd.uniform(-10, 10, size=(N, p))
        y_train = h_of_x1(X_train[:, 0]) + rnd.normal(0, sigma, size=N)

        model = get_model(model_name)
        model.fit(X_train, y_train)

        pred = model.predict(point)[0]
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_prediction = predictions.mean()
    
    variance = ((predictions - mean_prediction) ** 2).mean()
    bias = (mean_prediction - true_h) ** 2
    residual = sigma**2
    expected_error = bias + variance + residual

    return {
        "bias2": bias,
        "variance": variance,
        "residual": residual,
        "expected_error": expected_error,
        "mean_prediction": mean_prediction,
        "true_h": true_h,
    }


# ------------------
# -----TASK 2.2-----
# ------------------
# For the given models:
# 1. We create a test set of T points X_test whose coordinates x1 are evenly distributed over the interval (-10,10) 
#       and subsequent xi are randomly selected from a uniform distribution also over this interval
# 2. For each model, we train M independent models on different random training sets
# 3. We calculate the squared bias, variance, residual and expected error for each point
# 4. We plot the results as a function of the first feature x1
# 5. We compare the average predictions of the models with the Bayesian model (true h(x))

def compute_bias_variance(
    models,
    M=60,
    N=80,
    p=5,
    sigma=1.0,
    T=300,
    show_plots=True,
    out_dir="results",
    model_params=None
):
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(models, str):
        models = [models]

    # test set
    x1_test = np.linspace(-10, 10, T)
    X_test = np.zeros((T, p))
    X_test[:, 0] = x1_test
    X_test[:, 1:] = rnd.uniform(-10, 10, size=(T, p - 1))
    y_bayes = h_of_x1(x1_test)

    predictions = {name: np.zeros((M, T)) for name in models}

    # M independent models
    for m in range(M):
        X_train = rnd.uniform(-10, 10, size=(N, p))
        y_train = h_of_x1(X_train[:, 0]) + rnd.normal(0, sigma, size=N)

        for name in models:
            if model_params and name in model_params:
                params = model_params[name]
            else:
                if name == "ridge":
                    params= {"ridge":{"alpha":tune_parameter(name, candidate_values=[0.01, 0.1, 0.5, 1, 5, 10,15],N=N, p=p, sigma=sigma)}}
                elif name == "tree":
                    params= {"tree":{"max_depth":tune_parameter(name, candidate_values=[2,4,6,8,10,12,15,20,25,30],  N=N, p=p, sigma=sigma)}}
                elif name == "knn":
                    params= {"knn":{"k":tune_parameter(name, candidate_values=[1,2,3,4,5,7,9,11,13,15,19,21,23,30,40], N=N, p=p, sigma=sigma)}}
            
            model = get_model(name, **params)
            model.fit(X_train, y_train)
            predictions[name][m] = model.predict(X_test)

    results = {}
    for name in models:
        preds = predictions[name]
        mean_pred = preds.mean(axis=0)
        variance = ((preds - mean_pred)**2).mean(axis=0)
        bias2 = (mean_pred - y_bayes)**2
        residual = sigma**2
        error = bias2 + variance + residual

        results[name] = {
            "mean_prediction": mean_pred,
            "variance": variance,
            "bias2": bias2,
            "expected_error": error,
            "mean_error": error.mean(),
            "mean_bias2": bias2.mean(),
            "mean_variance": variance.mean(),
            "residual": residual
        }

    # optional plots
    if show_plots:
        fig, axes = plt.subplots(len(models), 1, figsize=(10, 4*len(models)), sharex=True)
        if len(models) == 1:
            axes = [axes]
        for ax, name in zip(axes, models):
            ax.plot(x1_test, results[name]["bias2"], label="bias²")
            ax.plot(x1_test, results[name]["variance"], label="variance")
            ax.plot(x1_test, [sigma**2]*T, label="residual")
            ax.plot(x1_test, results[name]["expected_error"], label="expected error")
            ax.set_title(name)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bias_variance_curves.png"))

        plt.figure(figsize=(10, 6))
        plt.plot(x1_test, y_bayes, label="Bayes", linewidth=2)
        for name in models:
            plt.plot(x1_test, results[name]["mean_prediction"], label=f"{name} avg")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "avg_preds_vs_bayes.png"))

    return results


# ------------------
# -----TASK 2.3-----
# ------------------
def learning_sample_size_effect_on_errors(models, N_values, M=60, p=5, sigma=1.0, T=300, out_dir="exp_N"):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # try for different learning sample sizes
    for N in N_values:
        for model_name in models:
            stats_dict = compute_bias_variance(
                models=[model_name],
                M=M, N=N, p=p, sigma=sigma, T=T, show_plots = False
            )
            stats = stats_dict[model_name] 
            stats["N"] = N
            stats["model"] = model_name
            results.append(stats)

    df = pd.DataFrame(results)

    # plot
    for model_name in models:
        sub = df[df["model"] == model_name]
        plt.figure(figsize=(8,5))
        plt.plot(sub["N"], sub["mean_bias2"], label="bias²")
        plt.plot(sub["N"], sub["mean_variance"], label="variance")
        plt.plot(sub["N"], np.full_like(sub["N"], sigma**2), label="residual")
        plt.plot(sub["N"], sub["mean_error"], label="expected_error")
        plt.title(f"{model_name}: mean error vs training size N")
        plt.xlabel("N")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{model_name}_vary_N.png"))

    return df


def complexity_effect_on_errors(model_name, complexity_values, M=60, N=80, p=5, sigma=1.0, T=300, out_dir="exp_complexity"):

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for value in complexity_values:
        if model_name == "ridge":
            model_params = {"ridge": {"alpha": value}}
        elif model_name == "knn":
            model_params = {"knn": {"k": value}}
        elif model_name == "tree":
            model_params = {"tree": {"max_depth": value}}
        
        # Relay model and parameters to compute_bias_variance
        stats_dict = compute_bias_variance(
            models=[model_name],
            M=M, N=N, p=p, sigma=sigma, T=T, 
            show_plots=False,
            model_params=model_params
        )

        stats = stats_dict[model_name]
        stats["complexity"] = value
        stats["model"] = model_name
        results.append(stats)


    df = pd.DataFrame(results)

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(df["complexity"], df["mean_bias2"], label="bias²")
    plt.plot(df["complexity"], df["mean_variance"], label="variance")
    plt.plot(df["complexity"], df["mean_error"], label="expected_error")
    plt.plot(df["complexity"], df["residual"], label="residual")
    plt.title(f"{model_name}: mean error vs model complexity")
    plt.xlabel("model complexity")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_vary_complexity.png"))

    return df



def irrevelant_variables_effect_on_errors(models, p_values, M=60, N=80, sigma=1.0, T=300, out_dir="exp_p"):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # try for different numbers of irrelevant variables
    for p in p_values:
        for model_name in models:
            stats_dict = compute_bias_variance(
                models=[model_name],
                M=M, N=N, p=p, sigma=sigma, T=T, show_plots = False
            )
            stats = stats_dict[model_name]
            stats["p"] = p
            stats["model"] = model_name
            results.append(stats)

    df = pd.DataFrame(results)

    # plot
    for model_name in models:
        sub = df[df["model"] == model_name]
        plt.figure(figsize=(8,5))
        plt.plot(sub["p"], sub["mean_bias2"], label="bias²")
        plt.plot(sub["p"], sub["mean_variance"], label="variance")
        plt.plot(sub["p"], np.full_like(sub["p"], sigma**2), label="residual")
        plt.plot(sub["p"], sub["mean_error"], label="expected_error")
        plt.title(f"{model_name}: mean error vs number of irrelevant vars p")
        plt.xlabel("p")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{model_name}_vary_p.png"))

    return df


# ------------------
# -----TASK 2.4-----
# ------------------
def compute_bias_variance_using_bagging(model_name, B_values, M=60, N=80, p=5, sigma=1.0, T=300):

    x1_test = np.linspace(-10, 10, T)
    X_test = np.zeros((T, p))
    X_test[:, 0] = x1_test
    X_test[:, 1:] = 0   # irrelevant variables set to 0
    y_true = h_of_x1(X_test[:, 0])

    results = []

    for B in B_values:
        predictions_M = np.zeros((M, T))  # bagging prediction for each LS

        for m in range(M):
         # generate one learning sample
            X_train = rnd.uniform(-10, 10, size=(N, p))
            y_train = h_of_x1(X_train[:, 0]) + rnd.normal(0, sigma, size=N)

            # create the base model
            base_model = get_model(model_name).model  # take raw sklearn model

            # bootstrap
            bag_model = BaggingRegressor(
                estimator=base_model,
                n_estimators=B,
                bootstrap=True,
                random_state=rnd.randint(0, 10000)
            )
            bag_model.fit(X_train, y_train)
            predictions_M[m] = bag_model.predict(X_test)

        mean_prediction = predictions_M.mean(axis=0)
        variance = ((predictions_M - mean_prediction)**2).mean(axis=0)
        bias2 = (mean_prediction - y_true)**2
        residual = sigma**2
        error = bias2 + variance + residual

        results.append({
            "B": B,
            "mean_bias2": bias2.mean(),
            "mean_variance": variance.mean(),
            "residual": residual,
            "expected_error": error.mean()
        })

    return pd.DataFrame(results)


# full bagging experiment
def show_bagging_results(models, B_values, M=60, N=80, p=5, sigma=1.0, T=300, out_dir="exp_bagging"):
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for model_name in models:
        df = compute_bias_variance_using_bagging(
            model_name, 
            B_values=B_values,
            M=M, N=N, p=p, sigma=sigma, T=T
        )

        all_results[model_name] = df

        # plot
        plt.figure(figsize=(8,5))
        plt.plot(df["B"], df["mean_bias2"], label="bias²")
        plt.plot(df["B"], df["mean_variance"], label="variance")
        plt.plot(df["B"], df["expected_error"], label="expected_error")
        plt.plot(df["B"], df["residual"], label="residual")
        plt.title(f"{model_name} – bagging effect")
        plt.xlabel("Number of bagged models B")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{model_name}_bagging.png"))

    return all_results



if __name__ == "__main__":
    results = compute_bias_variance(
        models=["ridge", "knn", "tree"],
        M=60,
        N=80,
        p=5,
        sigma=1.0,
        T=300,
        show_plots = True
    )

    #bias/variance in a given point
    x = [2.0, 0, 0, 0, 0]
    bv = compute_bias_variance_at_x("ridge", x)
    print("Bias/variance at x =", x, ":", bv)


    model_list = ["ridge", "knn", "tree"]

    df_N = learning_sample_size_effect_on_errors(model_list, N_values=[20,50,80,90,200,500,1000,1500,2000,2500,3500,6000])

    df_knn = complexity_effect_on_errors("knn", [1,3,5,7,9,11,13,15,17,19,21,23,25,30,35,40])
    df_ridge = complexity_effect_on_errors("ridge", [0.1,0.5,1,5,10,15,20,30,40,50,60,70,80,90,100])
    df_tree  = complexity_effect_on_errors("tree", [2,4,6,8,10,15,20,30,40])

    df_p = irrevelant_variables_effect_on_errors(model_list, p_values=[2,5,10,15,20,30,40,50,70,90])

    B_values = [1, 2, 4, 6,8,10,15,20,25,30,35,40,45,50]

    results_bagging = show_bagging_results(model_list, B_values)
