import subprocess
import sys
import time

notebooks = [
    "01_EDA_Analysis.ipynb",
    "02_Feature_Engineering.ipynb",
    "03_Clustering_Analysis.ipynb",
    "04_Classification_Models.ipynb",
    "05_Regression_Models.ipynb",
    "06_Hyperparameter_Tuning.ipynb",
    "07_Final_Comparison_MLflow.ipynb",
]

for nb in notebooks:
    print(f"\n{'='*50}")
    print(f"Execution : {nb}")
    print(f"{'='*50}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook",
         "--execute",
         "--inplace",
         "--ExecutePreprocessor.timeout=600",
         nb],
        capture_output=True, text=True
    )

    duration = time.time() - start

    if result.returncode == 0:
        print(f"OK — termine en {duration:.1f}s")
    else:
        print(f"ERREUR dans {nb}")
        print(result.stderr)
        print("Arret du pipeline.")
        sys.exit(1)

print("\nTous les notebooks ont ete executes avec succes.")