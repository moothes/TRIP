TRIP
=================
The source code of our paper: "Development and validation of an artificial intelligence system for triple-negative breast cancer identification and prognosis prediction: a multicentre retrospective study".

### Preparing data

You can prepare your pathology data following the steps described in [this link](https://github.com/mahmoodlab/SurvPath).   
After that, your should generate a ```.csv``` file to include the labels (TNBC/DFS/OS) and path to corresponding ```.pt``` file for the patients.

### Running 
```python
# Training
python main.py TRIP --task=[os/dfs/bcls]

# Test-time adaptation
python tta.py TRIP --task=[os/dfs/bcls] --test_set=[cohort_name]

# Testing
python test.py TRIP --task=[os/dfs/bcls]
```
