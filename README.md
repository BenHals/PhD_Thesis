This repository contains the source code and experimentation scripts for the four systems proposed in my PhD thesis. Some datesets are not included, and must be downloaded separately from publically available sources.

Systems included:
- Advantage
- AiRStream
- FiCSUM
- SELeCT

# AdvantageMemManagement
When dealing with streams of data, classification systems must take into account changes in data or concept drift. For example, predicting the top products on amazon in real time may drift as season change. One mthod of dealing with such change is to build a model for each similar concept. These models can be stored and reused when similar conditions reappear, for example a `summer` model may be stored and reused every summer. 

Managing how these models are stored is an important open research problem. To store a model repository in memory, it becomes neccessary to delete or merge models. The decision of which models to delete or merge can have a large impact on the performance and explainibility of a system. In this project, we propose a new `Advantage` based policy to manage memory. This policy optimizes both performance and explainibility.

For futher details, please see the [journal paper](https://link.springer.com/article/10.1007/s10618-021-00736-w).
![Advantage of a model](https://github.com/BenHals/AdvantageMemManagement/blob/main/readme_img.png?raw=true)
# Implementation

The basic idea of advantage is to measure the benefit of storing and reusing a model compared to rebuilding a replacement model.
A finite state machine is proposed as an efficient method of storing the information needed to compute this measure online.

## Instructions to Run
1. Install packages in requirements.txt
2. Create a data directory. This will contain the data set to run classification systems on. 
3. Create data sets. The system expects a nested folder structure, with the directory names indicating parameters. The bottom level data sets are expected to be in ARFF format. The script `GenerateDatastreamFiles\generate_dataset.py` can be used to create the synthetic data sets used in the paper. The data directory to create files in should be passed using the `-d` commandline argument. Other arguments modify the data sets created, such as `-st` for the type of stream, `-nc` for the total number of concepts, `-hd` and `-ed` for the difficulty levels to use (e.g. the depth of the tree used to label observations in the TREE data set). These are set to the defaults used in the paper.
4. To run classifiers on datasets, use the `FSMAdaptiveClassifier\run_datastream_classifier.py` script. Set commandline arguments. Most are set to reasonable defaults. 
- Set the data directory to use using the `-d` command.
- `-mm` can be used to specify the memory management policy to use. Our proposed variants are: `rA, auc` and `score`.
- The system will detect all data set `ARFF` files and run the specified system on each, creating result files in the same directory following the naming scheme.
5. To compute result measures, run the `SysMemResultsGen\run_csv_results.py` script. This will 1) produce a .txt file alongside each result file containing relevant measures, but also consolidate all results into a `SysMemResultsGen\Mem_Manage_Results\save-name.pickle` file where `save-name` is specified using the `-sn` argument. 
- This pickle file can be plotted using the relevant `SysMemResultsGen\mml_plot_bw.py` or `SysMemResultsGen\noise_plot_bw.py` scripts, passing `save-name.pickle` using the `-f` command.

# Citation
This work is published in the Journal of Data Mining and Knowledge Discovery. Please cite the following paper:
`Halstead, Ben, Yun Sing Koh, Patricia Riddle, Russel Pears, Mykola Pechenizkiy, and Albert Bifet. "Recurring concept memory management in data streams: exploiting data stream concept evolution to improve performance and transparency." Data Mining and Knowledge Discovery (2021): 1-41.`

Or as Bibtex:

`@article{halstead2021recurring,
  title={Recurring concept memory management in data streams: exploiting data stream concept evolution to improve performance and transparency},
  author={Halstead, Ben and Koh, Yun Sing and Riddle, Patricia and Pears, Russel and Pechenizkiy, Mykola and Bifet, Albert},
  journal={Data Mining and Knowledge Discovery},
  pages={1--41},
  year={2021},
  publisher={Springer}
}`

# AirStream

This work is published in *Machine Learning* as:
Halstead, B., Koh, Y.S., Riddle, P. et al. Analyzing and repairing concept drift adaptation in data stream classification. Mach Learn (2021). https://doi.org/10.1007/s10994-021-05993-w

A data stream system to make predictions in non-stationary conditions and adapt to changes in unknown/unobserved features.
AirStream was specifically designed for inferring air quality level from recent readings from surrounding sensors.
This process is affected by changing conditions such as wind direction, wind speed, pollution source behavior changes and seasonality.
These changes must be adapted to in some way, even if no monitoring is availiable.

AirStream uses state-of-the-art data stream methods to detect and adapt to change in conditions it cannot observe.
Based on a foundation of detecting change and dynamically switching classifier, AirStream can also create a model of these changing conditions.
AirStream detects _concept drift_, a change in distribution on incoming data to identify changes in causal conditions.
By matching current stream conditions to past classifiers, AirStream selects the best classifier to use or builds a new one.
Under the hypothesis that a classifier will perform similarly given similar conditions, this selection provides information on current conditions.

![Air Pollution](https://github.com/BenHals/AirStream/raw/master/Poll_overlay.jpg)

## Instructions To Run
0. Install 
 - numpy
 - scikit-multiflow (We depend on the HoeffdingTree implementation).
 - tqdm (For progress bars).
 - utm (For working with latitude and longitude locations).

### Data Formatting
1. Each data set should be in a folder inside `RawData`, with the name of the folder being the unique ID of the data set, `$dataname$`.
 - This folder should contain a csv file named `$dataname$_full.csv`. The first column in this file should be a datetime, then a column for each sensor (feature) then a column for each auxiliary feature (used for evaluating relationship to weather, not for testing/training or deployment).
 - This folder should also contain a `$dataname$_sensors.json` file giving the x,y positions for each sensor in meters.
 - An example is shown in the `TestData` folder, and a jupyter notebook used to construct the files from raw data is given as `process_data_template.ipynb`.

### Running Evaluation
2. The main evaluation entry point is `conduct_experiment.py`. This script runs all baselines and AirStream on a specified data set with all combinations of passed settings. The script also preprocesses the data file, setting the target sensor, removing auxiliary features and applying masking.
 - Important command line arguments availiable are:
  - `-rwf`: The `$dataname$` of the data set, must match the folder and files in `RawData`.
  - `-hd`: A flag denoting if the data files have headers. Usually should be set.
  - `-ti`: The index of the sensor to designate the target, i.e. the sensor which will be predicted. Can specify multiple to run multiple times using different targets e.g. `-ti 0 1 2` will run 3 times using sensor 0, 1 and 2 as targets.
  - `-bp` & `-bl`: The break percentage and break length respectively (in terms of observations seen). Used for masking. Defaults to bp of 0.02 and bl of 75.
  - `-d`: changing the directory containing input. Should be set to a parent directory containing a `RawData` directory.
  - `-o`: changing the directory containing output. Should be set to a parent directory containing a `experiments` directory.
  - `-bcs`: Select which baselines to run from lin, temp, normSCG, OK, tree and arf. Defaults to all.
  - Arguments for changing AirStream settings. Refer to code, defaults to those used in the paper.
  - Output is written to `experiments` with a file structure: `experiments/$dataname$/$target index$/$seed$`
  - All classifier output is written to this folder, with AirStream results denoted `sys...`.
  - The `..._results.json` files provide all results for a given classifier, including accuracy and relationship to auxiliary features.
  - An example command to run is: `python conduct_experiment.py -rwf TestData -hd`

### Running AirStream
3. Code for the AirStream classifier is in `ds_classifier.py`. To use:
 - Initialize the classifier using `AirStream = DSClassifier(learner = HoeffdingTree)` (or pass in settings).
 - Incrementally train using `AirStream.partial_fit(X, y)` where X is a list of observations (use shape [1, -1] for a single observation) and y is a list of labels.
 - Classify using `AirStream.predict(X)` where X is a list of observations (use shape [1, -1] for a single observation).
 - AirStream active state at any point is given by `AirStream.active_state`. The value immediately prior to any prediction is the ID of the classifier used to make that prediction.


### Citing:
Please use the following citation:
`@article{halstead2021analyzing,
  title={Analyzing and repairing concept drift adaptation in data stream classification},
  author={Halstead, Ben and Koh, Yun Sing and Riddle, Patricia and Pears, Russel and Pechenizkiy, Mykola and Bifet, Albert and Olivares, Gustavo and Coulson, Guy},
  journal={Machine Learning},
  pages={1--35},
  year={2021},
  publisher={Springer}
}`


# FiCSUM
A data stream framework to make predictions in non-stationary conditions.
FiCSUM uses a combination of many "meta-information" features in order to detect change in many aspects of a data stream, both supervised (change in relationship to labels) and unsupervised (change in feature space).
Individually, each meta-information feature has been shown to be able to discriminate between useful "concepts" in a data stream, or periods displaying a similar relationship to be learned. 
In real-world streams change is possible in many different aspects, and single meta-information features are not able to detect all aspects at once.
For example, looking only at feature space may miss changes in class label distribution.
FiCSUM shows that a combined approach can increase performance.

For more details, please see the full paper published in ICDE 2021 availiable here.

![Concept Similarity](https://github.com/BenHals/PhDCode/raw/master/ConceptSimilarity.png)
# Implementation

The basic idea of FiCSUM is to capture many aspects of data stream behaviour in a vector, which we call a 'concept fingerprint'.
This fingerprint represents an overall set of behaviours seen over a window of observations. 
Vector similarity measures can be used to compare the fingerprints at different points in the stream.
This allows change to be detected, as a significant difference in similarity, and reccurences to be identified, as a resurgence in similarity to a previously seen fingerprint.
This allows change to be adapted to by saving an individual model to be used along side each fingerprint, and reused when it best matches the stream.
This also allows a concept history of the stream to be built, representing similar periods in the stream. A concept history can be mapped to past behaviour and even environmental conditions to contextualize future recurrence. For example, if a previous fingerprint was associated with medium accuracy and always occured alongside stormy conditions, a future recurrence to this fingerprint can be expected to also bring medium accuracy and stormy conditions.

## Instructions to Run
0. Install
- numpy
- scikit-multiflow
- tqdm
- psutil
- statsmodels
- pyinstrument
1. meta-information dependencies
- [shap](https://github.com/slundberg/shap) version 0.35 (version is required as we use a patched file based on this version. Will require the patched code to be updated for a newer version.)
- Enable shap to work with Scikit-Multiflow by replacing the shap tree.py file with the patched `tree.py` found in `PhDCode\Exploration`. (Or just use the section defining the translation from )
- [PyEMD](https://github.com/laszukdawid/PyEMD)
- [EntroPy](https://raphaelvallat.com/entropy/build/html/index.html)
- Can run pip -install -r requirements.txt
2. Create a data directory. This is expected to contain `Real` and `Synthetic` subdirectories. These should each contain a directory for each data set to be tested on. The expected format for these is a `.csv` file for each ground truth context. The system will work with only a single `.csv` if context is unknown, but some evaluation measures will not be able to be calculated. For synthetic datasets created with a known generator, an empty directory in the `Synthetic` directory is needed to store files. Each dataset folder should be named the name you will use to call it. The base data directory should be passed as the `--datalocation` commandline argument. The dataset name is passed in the `--datasets` argument. New datasets will need to be added to the relevent list of allowed datasets in `run_experiment.py`, `synthetic_MI_datasets` for `Synthetic` datasets, or `real_drift_datasets` for `Real` datasets.
3. Set commandline arguments. Most are set to reasonable defaults. 
- Set seed using the `--seeds` command. This should be set to the number of runs if `--seedaction` is set to `new` (will create new random seeds) or `reuse` (will reuse previously used seeds). Or should be a list if `--seedaction` is `list`, e.g. `--seedaction list --seeds 1 2 3` will run 3 runs using seeds 1, 2 and 3.
- Set multiprocessing options. `--single` can be set to turn off multiprocessing for simpler output and ease of cancelation. Or `--cpus` can set the desired number of cpu cores to run on.
- Set meta-information functions and behaviour sources to disable using `--ifeatures` and `--isources`. For Quick runs, disabling `MI` and `IMF` can improve runtime significantly.
4. Results are placed in `~\output\expDefault\[dataset_name]` by default. This can be set with `--outputlocation` and `--experimentname`.
- The `results_run_....txt` file contains the calculated performance measures of the run.
- The `run_...csv` file contains measures describing how each observation was handled.
- The `run_...._options.txt` file contains the options used for a specific run.


### Running
The main evaluation entry point is `run_experiment.py`. This script runs FiCSUM on a specified data set. 
`run_moa.py` runs the same experiment calling a moa classifier on the commandline. Used to test against baselines.
`run_other.py` runs the same experiment using alternative python-based frameworks. Used to test against baselines.

# Citation
Please cite this work as:
`Halstead, Ben, Yun Sing Koh, Patricia Riddle, Russel Pears, Mykola Pechenizkiy, and Albert Bifet. "Fingerprinting Concepts in Data Streams with
Supervised and Unsupervised Meta-Information
" International Conference on Data Engineering (ICDE) (2021)`

# SELeCT
Learning from streaming data requires handling changes in distribution, known as concept drift, in order to maintain performance. Adaptive learning approaches retain performance across concept drift by explicitly changing the classifier used to handle incoming observations, allowing changes in distribution to be tracked and adapted to.However, previous methods fail to select the optimal classifier in many common scenarios due to sparse evaluations of stored classifiers, leading to reduced performance. We propose a probabilistic framework, SELeCT, which is able to avoid these failure cases by continuously evaluating all stored classifiers.

The SELeCT framework uses a Bayesian approach to assign each classifier a probability of representing a similar distribution to incoming data, combining a prior probability based on the current state of the system with the likelihood of drawing recent observations. A continuous selection procedure based on the Hoeffding bound ensures that each observation is classified by the classifier trained on the most similar distribution. SELeCT achieves accuracies up to 7% higher than the standard framework, with  classifier use matching ground truth concept dynamics with over 90% recall and precision. 

This figure shows how the probabilities of each classifier change over time. Notice that recurring concepts trigger resurgences in the probability of the classifier used on the last occurence.
![State Probabilities](https://github.com/BenHals/SELeCT/blob/master/AD-posterior.png)
# Implementation

The SELeCT framework retains the representation of concepts as states, and the overall aim of selecting the optimal state to handle each observation, but reformulates drift detection and re-identification as a continuous, probabilistic selection procedure. SELeCT solves the issue of sparse evaluations of states by providing a full probability distribution for all states at every observation and guarantees that the selected state is the optimal achievable state.

For each state, SELeCT computes the probability that the state is optimal for the next observation using a Bayesian framework. A state prior based on current knowledge of the system is combined with a state likelihood based on recent observations. Drift detection is integrated into the prior probability component along with other sources of prior knowledge, such as the current active state. Re-identification is reformulated as calculating the likelihood of drawing recent observations from each state. These components, along with a continuous selection procedure, allow SELeCT to guarantee, within some error bound, that the selected state is the optimal achievable state under a given similarity measure.

## Instructions to Run
0. Using python 3.7, install requirements using `pip install -r requirements.txt`
- Install SELeCT module with `pip install -e .` run inside the SELeCT-master directory.
1. (Optional) Place desired datasets into the `RawData` directory. This is expected to contain `Real` and `Synthetic` subdirectories. These should each contain a directory for each data set to be tested on. The expected format for these is a `.csv` file for each ground truth context. The system will work with only a single `.csv` if context is unknown, but some evaluation measures will not be able to be calculated. For synthetic datasets created with a known generator, an empty directory in the `Synthetic` directory is needed to store files. Each dataset folder should be named the name you will use to call it. The base data directory should be passed as the `--datalocation` commandline argument. The dataset name is passed in the `--datasets` argument. New datasets will need to be added to the relevent list of allowed datasets in `run_experiment.py`, `synthetic_MI_datasets` for `Synthetic` datasets, or `real_drift_datasets` for `Real` datasets.
2. Run code for demos, discussed below.
3. Run main entry point, `run_experiment.py`. Set commandline arguments. Most are set to reasonable defaults. 
- Set seed using the `--seeds` command. This should be set to the number of runs if `--seedaction` is set to `new` (will create new random seeds) or `reuse` (will reuse previously used seeds). Or should be a list if `--seedaction` is `list`, e.g. `--seedaction list --seeds 1 2 3` will run 3 runs using seeds 1, 2 and 3.
- Set multiprocessing options. `--single` can be set to turn off multiprocessing for simpler output and ease of cancelation. Or `--cpus` can set the desired number of cpu cores to run on.
- Set meta-information functions and behaviour sources to disable using `--ifeatures` and `--isources`. For Quick runs, disabling `MI` and `IMF` can improve runtime significantly.
4. Results are placed in `~\output\expDefault\[dataset_name]` by default. This can be set with `--outputlocation` and `--experimentname`.
- The `results_run_....txt` file contains the calculated performance measures of the run.
- The `run_...csv` file contains measures describing how each observation was handled.
- The `run_...._options.txt` file contains the options used for a specific run.


### Running
The main evaluation entry point is `run_experiment.py`. This script runs SELeCT on a specified data set. 
`run_experiment_moa.py` runs the same experiment calling a moa classifier on the commandline. Used to test against baselines.

The `demo` directory gives an example of recreating results from the paper. To recreate the results on the CMC dataset:
1. Run the `demo\cmc_testbed.py` file. This will run `run_experiment.py` with the correct settings to recreate the results from the paper.
- The test by default runs on 45 seeds. To speed up the test using parallel processing, pass the commandline argument `--cpus X` where X is the number of CPU cores you wish to use.
2. The results of the test will print once done. The individual results for each seed can be found in `output\cmc_testbed\**`.
3. The `--dataset` commandline argument can be passed to run other tests. The other realworld datasets will need the appropriate dataset to be downloaded. Synthetic datasets can be run following the above instructions.



### Wind simulation demos
We also include a demo of SELeCT running on our air quality simulation. 
1. Run the `demo\windSimStream.py` file. A GUI should appear showing a view of the system over time.

![Windsim Demo](https://github.com/BenHals/SELeCT/blob/master/wind_demo.gif)
