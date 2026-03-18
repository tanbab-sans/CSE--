import faulthandler
faulthandler.enable()
import numpy as np
import torch

def compute_global_score(
    test_result,
    ood_result,
    test_mean_simulation_time,
    test_ood_mean_simulation_time,
    speedup_zero=False,
):
    import math
    thresholds = {
        "x-velocity":(0.01,0.02,"min"),
        "y-velocity":(0.01,0.02,"min"),
        "pressure":(0.002,0.01,"min"),
        "pressure_surfacic":(0.008,0.02,"min"),
        "turbulent_viscosity":(0.05,0.1,"min"),
        "mean_relative_drag":(0.4,5.0,"min"),
        "mean_relative_lift":(0.1,0.3,"min"),
        "spearman_correlation_drag":(0.8,0.9,"max"),
        "spearman_correlation_lift":(0.96,0.99,"max")
    }

    configuration = {
        "coefficients": {"ML": 0.4, "OOD": 0.3, "Physics": 0.3},
        "ratioRelevance": {"Speed-up": 0.25, "Accuracy": 0.75},
        "valueByColor": {"g": 2, "o": 1, "r": 0},
        "maxSpeedRatioAllowed": 10000,
        "reference_mean_simulation_time": 1500,
    }

    ml_metrics = test_result["test"]["ML"]["MSE_normalized"]
    ml_metrics["pressure_surfacic"] = test_result["test"]["ML"][
        "MSE_normalized_surfacic"
    ]["pressure"]

    phy_variables_to_keep = [
        "mean_relative_drag",
        "mean_relative_lift",
        "spearman_correlation_drag",
        "spearman_correlation_lift",
    ]
    phy_metrics = {
        phy_variable: test_result["test"]["Physics"][phy_variable]
        for phy_variable in phy_variables_to_keep
    }

    ml_ood_metrics = ood_result["test_ood"]["ML"]["MSE_normalized"]
    ml_ood_metrics["pressure_surfacic"] = ood_result["test_ood"]["ML"][
        "MSE_normalized_surfacic"
    ]["pressure"]
    phy_ood_metrics = {
        phy_variable: ood_result["test_ood"]["Physics"][phy_variable]
        for phy_variable in phy_variables_to_keep
    }
    ood_metrics = {**ml_ood_metrics, **phy_ood_metrics}

    all_metrics = {"ML": ml_metrics, "Physics": phy_metrics, "OOD": ood_metrics}

    print("---------------------------------")
    print("Global Score Computation")
    print()
    print("All Metrics")
    print("ML", ml_metrics)
    print("Physics", phy_metrics)
    print("OOD", ood_metrics)


    reference_mean_simulation_time = configuration["reference_mean_simulation_time"]
    speedUp = {
        "ML": reference_mean_simulation_time / test_mean_simulation_time,
        "OOD": reference_mean_simulation_time / test_ood_mean_simulation_time,
    }

    accuracyResults = dict()
    for subcategoryName, subcategoryVal in all_metrics.items():
        accuracyResults[subcategoryName] = []
        for variableName, variableError in subcategoryVal.items():
            thresholdMin, thresholdMax, evalType = thresholds[variableName]
            if evalType == "min":
                if variableError < thresholdMin:
                    accuracyEval = "g"
                elif thresholdMin < variableError < thresholdMax:
                    accuracyEval = "o"
                else:
                    accuracyEval = "r"
            elif evalType == "max":
                if variableError < thresholdMin:
                    accuracyEval = "r"
                elif thresholdMin < variableError < thresholdMax:
                    accuracyEval = "o"
                else:
                    accuracyEval = "g"

            accuracyResults[subcategoryName].append(accuracyEval)

    print("accuracyResults: ", accuracyResults)

    string_ = ""
    string_ += "ML: "+str(ml_metrics)+"\n"
    string_ += "Physics: "+str(phy_metrics)+"\n"
    string_ += "OOD: "+str(ood_metrics)+"\n"
    string_ += "accuracyResults: "+str(accuracyResults)+"\n"

    if speedup_zero:
        speedUp["ML"] = 1e-10
        speedUp["OOD"] = 1e-10
    print("speedUp: ", speedUp)

    coefficients = configuration["coefficients"]
    ratioRelevance = configuration["ratioRelevance"]
    valueByColor = configuration["valueByColor"]
    maxSpeedRatioAllowed = configuration["maxSpeedRatioAllowed"]
    mlSubscore = 0
    accuracyMaxPoints = ratioRelevance["Accuracy"]
    accuracyResult = sum([valueByColor[color] for color in accuracyResults["ML"]])
    accuracyResult = (
        accuracyResult
        * accuracyMaxPoints
        / (len(accuracyResults["ML"]) * max(valueByColor.values()))
    )
    mlSubscore += accuracyResult
    print("ML accuracyResult", accuracyResult)

    speedUpMaxPoints = ratioRelevance["Speed-up"]
    speedUpResult = max(
        0, min(math.log10(speedUp["ML"]) / math.log10(maxSpeedRatioAllowed), 1)
    )
    speedUpResult = speedUpResult * speedUpMaxPoints
    mlSubscore += speedUpResult
    print("ML speedUpResult", speedUpResult)

    accuracyResult = sum([valueByColor[color] for color in accuracyResults["Physics"]])
    accuracyResult = accuracyResult / (
        len(accuracyResults["Physics"]) * max(valueByColor.values())
    )
    physicsSubscore = accuracyResult
    print("Phy accuracyResult", accuracyResult)

    oodSubscore = 0
    accuracyMaxPoints = ratioRelevance["Accuracy"]
    accuracyResult = sum([valueByColor[color] for color in accuracyResults["OOD"]])
    accuracyResult = (
        accuracyResult
        * accuracyMaxPoints
        / (len(accuracyResults["OOD"]) * max(valueByColor.values()))
    )
    oodSubscore += accuracyResult
    print("OOD accuracyResult", accuracyResult)

    speedUpMaxPoints = ratioRelevance["Speed-up"]
    speedUpResult = max(
        0, min(math.log10(speedUp["OOD"]) / math.log10(maxSpeedRatioAllowed), 1)
    )
    speedUpResult = speedUpResult * speedUpMaxPoints
    oodSubscore += speedUpResult
    print("OOD speedUpResult", speedUpResult)

    globalScore = 100 * (
        coefficients["ML"] * mlSubscore
        + coefficients["Physics"] * physicsSubscore
        + coefficients["OOD"] * oodSubscore
    )

    return globalScore, string_


if __name__ == "__main__":

    import os, time
    from lips import get_root_path
    from lips.dataset.airfransDataSet import download_data, extract_dataset_by_simulations
    from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
    from safran_augmented_simulation import AugmentedSimulator

    LIPS_PATH = get_root_path()
    DIRECTORY_NAME = 'Dataset'
    BENCHMARK_NAME = "Case1"
    LOG_PATH = LIPS_PATH + "lips_logs.log"

    BENCH_CONFIG_PATH = os.path.join("airfoilConfigurations","benchmarks","confAirfoil.ini") #Configuration file related to the benchmark
    SIM_CONFIG_PATH = os.path.join("airfoilConfigurations","simulators","torch_fc.ini")

    import os
    if not os.path.isdir(DIRECTORY_NAME):
        download_data(root_path=".", directory_name=DIRECTORY_NAME)

    benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                                config_path = BENCH_CONFIG_PATH,
                                benchmark_name = BENCHMARK_NAME,
                                log_path = LOG_PATH)
    benchmark.load(path=DIRECTORY_NAME)
    

    simulator = AugmentedSimulator(benchmark)

    start = time.time()
    simulator.train(benchmark.train_dataset)
    print(f"time to train = {time.time() - start}")

    ####################

    start_test = time.time()
    fc_metrics_test = benchmark.evaluate_simulator(
        dataset="test", augmented_simulator=simulator
    )
    print("fc_metrics_test =", fc_metrics_test)
    test_evaluation_time = time.time() - start_test
    test_mean_simulation_time = test_evaluation_time / len(
        benchmark._test_dataset.get_simulations_sizes()
    )
    print("Test evaluation time:", test_evaluation_time)

    start_test = time.time()
    fc_metrics_test_ood = benchmark.evaluate_simulator(
        dataset="test_ood", augmented_simulator=simulator
    )
    print("fc_metrics_test_ood =", fc_metrics_test_ood)
    test_evaluation_time = time.time() - start_test
    test_ood_mean_simulation_time = test_evaluation_time / len(
        benchmark._test_ood_dataset.get_simulations_sizes()
    )
    print("Test_ood evaluation time:", test_evaluation_time)

    ####################

    score, string = compute_global_score(
        fc_metrics_test,
        fc_metrics_test_ood,
        test_mean_simulation_time,
        test_ood_mean_simulation_time,
        speedup_zero=False,
    )

    output_string = ""
    output_string += "nb_modes_shape_embedding : "+str(simulator.nb_modes_shape_embedding)+"\n"
    output_string += "nb_modes_fields : "+str(simulator.nb_modes_fields)+"\n"
    output_string += "-------------------------------------------------------------\n"
    output_string += string
    output_string += "-------------------------------------------------------------\n"
    output_string += "globalScore = "+str(score)+"\n"

    print(output_string)
