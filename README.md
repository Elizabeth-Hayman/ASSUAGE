---
# ASSUAGE : **AS**ssisted **SU**rrogate **A**ided **G**enerative **E**ngineering
---

Requires user input of a reduced and a full model. Both must contain a run.sh script which executes a simulation


Alpha causing problems - how to user specify if multiple sims needed one point??
Can we read extra arguments into encoding surrogate

Suggest installing as pip install -e . 
This allows any edits in the base code to directly act

# ASSUAGE Execution Workflow  

This document explains, at a high level and in user-friendly terms, the workflow demonstrated by the example execution script in this folder. It covers:

- How the parameter bounds and template folders are used  
- What the **Data Exploration** step does  
- What the **Machine Learning Model Fitting** step does  
- What files are created and where they are stored  
- How to run and extend the workflow  
- General installation advice  

This README is written for users with limited coding background.

---

# 1. Overview

The ASSUAGE workflow uses a combination of:

- **Geometry encoding functions** (`parameter_to_model`, `preprocess_parameters`)  
- **Ground-truth CFD data generation** (optional)  
- **Data investigation & quality checks**  
- **Surrogate model training using ML**  
- **Final surrogate pipeline export**  

The example script shows how to:

1. Define the design parameter space  
2. (Optionally) generate new ground-truth CFD runs  
3. Explore the dataset visually  
4. Run a full suite of ML models  
5. Automatically select the best model  
6. Save and test the final surrogate pipeline  

Everything is designed to be automated and easy to repeat.

---

# 2. Parameter Setup

The script defines lower and upper bounds for **24 design variables**, grouped as:

- `radius[0..5]`
- `zPos[0..5]`
- `number[0..5]`
- `offset[0..5]`

These bounds control the geometry generator (`parameter_to_model`).  
They ensure that all random designs used during dataset creation or optimisation stay within safe, physically meaningful ranges.

You should adjust these bounds if your design space changes.

---

# 3. Encoding Functions

Two key functions come from `encoding.py`:

### ✔ `parameter_to_model(values, folder, alpha)`
Generates a full simulation folder from a vector of design parameters.

### ✔ `preprocess_parameters(values)`
Checks if a parameter vector is valid before running expensive simulations  
(e.g., ensures hole overlap is not too large, z-length is not excessive).

If `preprocess_parameters` is missing, the script continues without filtering.

These functions are essential when generating new CFD datasets or running optimisation.

---

# 4. Ground-Truth Data Generation (optional)

This section is commented out in the example:

```python
from ASSUAGE.create_ground_truth import start_new_runs, extractData
start_new_runs(...)
extractData(extract_surrogate_inputs, extract_fitness)
```

If enabled, it would:

1. Randomly sample parameter vectors within the defined bounds  
2. Generate simulation folders  
3. Run full CFD simulations  
4. Extract surrogate inputs (reduced model outputs)  
5. Extract fitness values (full model outputs)  
6. Build a dataset under `surrogateCreation/`  

Most users only do this once.

---

# 5. Data Investigation (optional but recommended)

The script includes a block (commented out) using:

```python
from ASSUAGE.surrogateModel.data_exploration import data_exploration
```

This tool performs **data quality and structure analysis**:

### ✔ Correlation matrix  
Shows how strongly features relate.

### ✔ Feature histograms  
Checks distributions.

### ✔ PCA reduction & explained variance  
Shows dimensionality of the data.

### ✔ Pairplots  
Visual pairwise relationships.

### ✔ Outlier detection  
Flags anomalous samples.

### ✔ Proxy feature importance  
Rough estimate from a small Random Forest.

All results are saved in `surrogateCreation/`.

This step verifies whether your dataset looks healthy before training ML models.

---

# 6. Machine Learning Model Fitting

The core of the example script is:

```python
from ASSUAGE.surrogateModel.mlModelFitting import mlModels
```

You create an instance like:

```python
modeler = mlModels(
    input_data="surrogateCreation/trainingInput.csv",
    output_data="surrogateCreation/trainingOutput.csv",
    ...
)
```

This class:

- Loads your dataset  
- Builds a pipeline:  
  ```
  StandardScaler → (optional PCA) → Model
  ```
- Tries many regression models  
- Tunes their hyperparameters  
- Evaluates them using cross-validation  
- Tracks the best model *automatically*  
- Saves a final “best model” pipeline with pickle  

### Running model comparison

```python
results = modeler.evaluate_nested_cv(
    outer_splits=5,
    inner_splits=5,
    create_plots=True
)
```

This performs **nested cross-validation**:

- Inner loop → hyperparameter tuning  
- Outer loop → unbiased performance evaluation  

Plots, JSON, CSVs, and predictions are saved in `surrogateCreation/`.

### Inspecting best model

```python
print(modeler.best_model_name)
print(modeler.best_model_score)
print(modeler.best_model_info)
```

The toolkit automatically records the best model while evaluations run.

---

# 7. Final Surrogate Model Training

After model comparison, run:

```python
pipeline = modeler.train_best_pipeline(cv=5)
```

This step:

- Selects the best model recorded earlier  
- Performs a clean CV hyperparameter search on **all** of the data  
- Retrains the full pipeline  
- Saves it to disk (e.g., `demo_MLP_best_pipeline.pkl`)  
- Makes it available in Python via:
  ```python
  modeler.best_pipeline
  modeler.best_pipeline_path
  ```

---

# 8. Using the Saved Pipeline

The saved model can be loaded using `pickle`:

```python
with open(modeler.best_pipeline_path, "rb") as fh:
    loaded_pipe = pickle.load(fh)

sample_X = pd.read_csv("surrogateCreation/trainingInput.csv", header=None).iloc[:5].to_numpy()
preds = loaded_pipe.predict(sample_X)
```

This allows you to use the surrogate model for:

- Parameter studies  
- Optimisation loops  
- Fast approximations of the CFD model  

---

# 9. Optional: Feature Importance Visualisation

The toolkit can also compute and plot feature importances:

```python
df = modeler.save_and_plot_feature_importance()
```

Features are ranked and plotted.  
If PCA is used, importance is mapped back to original features.

Results are saved in `surrogateCreation/`.

---

# 10. Output Directory Structure

After running the script, you will typically see:

```
surrogateCreation/
    trainingInput.csv
    trainingOutput.csv
    demo*best_pipeline.pkl
    demo*_results_*.json
    demo*_summary_*.csv
    *predictions.csv
    *summary.png
    feature_importances_*.csv
    feature_importances_*.png
```

Everything needed for reporting, analysis, and downstream optimisation is stored here.

---

# 11. Installation & Execution

## Recommended installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn
```

Ensure your Python environment is consistent across runs when saving and loading pickle files.

## Running the script

```bash
python execution.py
```

Ground-truth creation and data exploration blocks are optional and can be uncommented as needed.

---

# 12. Summary

This example script demonstrates the full ASSUAGE surrogate-model workflow:

1. Define parameter bounds  
2. (Optional) generate ground-truth data  
3. Explore and validate the dataset  
4. Evaluate many ML models automatically  
5. Automatically select the best model  
6. Retrain it on all data  
7. Save the final surrogate  
8. Use it for prediction, optimisation, or deployment  

The workflow is designed to be **easy**, **robust**, and **repeatable**.

For questions or extensions (adding new models, integrating optimisation loops, or custom plotting), feel free to ask!
