import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\dlals\Desktop\NeurIPS2024-ML4CFD-competition-Starting-Kit-main"
MMGP_DIR = os.path.join(BASE_DIR, "Winning-solutions-Previous-Challenge", "Rank 1st-MMGP")

sys.path.insert(0, MMGP_DIR)

from safran_augmented_simulation import AugmentedSimulator
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

def main():
    print("Loading benchmark...")
    benchmark = AirfRANSBenchmark(
        benchmark_path=os.path.join(BASE_DIR, "Dataset"),
        config_path=os.path.join(BASE_DIR, "airfoilConfigurations", "benchmarks", "confAirfoil.ini"),
        benchmark_name="Case1",
        log_path="lips_logs.log"
    )
    print("Benchmark object created, calling load...")
    benchmark.load(path=os.path.join(BASE_DIR, "Dataset"))
    print("Benchmark load complete.")

    print("Instantiating Simulator...")
    sim = AugmentedSimulator(benchmark=benchmark)

    print("Training Simulator... this will process datasets and fit GPs")
    # safran_augmented_simulation handles the GP fitting in train
    sim.train(benchmark.train_dataset)

    print("Evaluating Simulator on Test...")
    start_test = time.time()
    metrics_test = benchmark.evaluate_simulator(dataset="test", augmented_simulator=sim, eval_batch_size=256000, num_workers=1)
    test_eval_time = time.time() - start_test
    print("Test Time:", test_eval_time)
    print("Test Metrics:")
    print(metrics_test)

    print("Evaluating Simulator on Test OOD...")
    start_ood = time.time()
    metrics_ood = benchmark.evaluate_simulator(dataset="test_ood", augmented_simulator=sim, eval_batch_size=256000, num_workers=1)
    ood_eval_time = time.time() - start_ood
    print("OOD Time:", ood_eval_time)
    print("OOD Metrics:")
    print(metrics_ood)
    
    # Save the metrics to a json to calculate score later
    results = {
        "test": metrics_test["test"]["ML"],
        "test_phys": metrics_test["test"]["Physics"],
        "ood": metrics_ood["test_ood"]["ML"],
        "ood_phys": metrics_ood["test_ood"]["Physics"],
        "speedUp": {
            "ML": (len(benchmark._test_dataset.get_simulations_sizes()) * 100) / test_eval_time,
            "OOD": (len(benchmark._test_ood_dataset.get_simulations_sizes()) * 100) / ood_eval_time
        }
    }
    with open("mmgp_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved mmgp_results.json")

    print("Generating Visualizations...")
    
    # Extract the first simulation from the OOD set for visualization
    ood_dataset = benchmark._test_ood_dataset
    sizes = ood_dataset.get_simulations_sizes()
    first_sim_size = sizes[0]
    
    # Ground Truth data
    points = ood_dataset.data['x-position'][:first_sim_size]
    y_points = ood_dataset.data['y-position'][:first_sim_size]
    true_x_vel = ood_dataset.data['x-velocity'][:first_sim_size]
    true_y_vel = ood_dataset.data['y-velocity'][:first_sim_size]
    true_pressure = ood_dataset.data['pressure'][:first_sim_size]

    # Model Predictions
    preds = sim.predict(ood_dataset)
    pred_x_vel = preds['x-velocity'][:first_sim_size]
    pred_y_vel = preds['y-velocity'][:first_sim_size]
    pred_pressure = preds['pressure'][:first_sim_size]

    # Create plot for Pressure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(points, y_points, c=true_pressure, cmap='jet', s=1)
    axes[0].set_title('Ground Truth: Pressure')
    axes[0].set_aspect('equal')
    fig.colorbar(sc1, ax=axes[0])
    
    sc2 = axes[1].scatter(points, y_points, c=pred_pressure, cmap='jet', s=1)
    axes[1].set_title('MMGP Prediction: Pressure')
    axes[1].set_aspect('equal')
    fig.colorbar(sc2, ax=axes[1])
    
    plt.savefig('pressure_comparison.png', dpi=300)
    plt.close()

    # Create plot for Velocity Magnitude 
    true_vel_mag = np.sqrt(true_x_vel**2 + true_y_vel**2)
    pred_vel_mag = np.sqrt(pred_x_vel**2 + pred_y_vel**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(points, y_points, c=true_vel_mag, cmap='viridis', s=1)
    axes[0].set_title('Ground Truth: Velocity Magnitude')
    axes[0].set_aspect('equal')
    fig.colorbar(sc1, ax=axes[0])
    
    sc2 = axes[1].scatter(points, y_points, c=pred_vel_mag, cmap='viridis', s=1)
    axes[1].set_title('MMGP Prediction: Velocity Magnitude')
    axes[1].set_aspect('equal')
    fig.colorbar(sc2, ax=axes[1])
    
    plt.savefig('velocity_comparison.png', dpi=300)
    plt.close()
    
    print("Visualizations saved as pressure_comparison.png and velocity_comparison.png")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException as e:
        print("CRITICAL ERROR IN MAIN:", type(e), e)
        traceback.print_exc()
        sys.exit(1)
