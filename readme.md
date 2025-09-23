# Object detection for BDD100k dataset

The repository is structured into 3 folders, as per the 3 segements from the assignment. 
- **part1_analysis** folder contains the main script `analysis_new.py` for data analysis that produces the stats and visualisations, along with dockerfile and instructions to run them. Please go through the `readme.md` file that shows the observations and other details of data analysis. The artifacts generated area already a part of the repository and under bdd100k_analysis_train and val folders respectively. 

- **part2_model** contains a few items: 
    - a folder called `pretrained_model_inference` that has python scripts and notebooks for running inference on selected pre-trained models on the model zoo,
    - one folder called **yolo_training** that contains:
        1. script `convert_bdd_yolo.py` to convert the bdd100k dataset yo yolo format to make it training compatible,
        2. `train_yolo.py` which is the training script with the initial set of hyperparams chosen for this task,
        3. `bdd100k_training` folder which contains the trained models along with training metrics. 
    - also, a report called `pretrained_models_report.pdf` that is generated using the assitance of Claude and inputs from my end, which is meant to serve as a reference guide on information about most of the models present on the model zoo and what their pros/cons for this task might look like, to make it easy for us to refer to. 
    - `readme.md` file that contains the findings and observations of the chosen models.


    NOTE:
    One thing to mention here is that the pre-trained model inference script uses `mmcv` package, as the official guides suggest, but there were several issues found in getting this compatible and working as noted by other users on github (mainly the package not working well with specific versions of cuda and pytorch).


- **part3_evals** contains the quantitaive and qualitative evaluation results of the models. 
    1. `readme.md` contains the comprehensive evaluation along with what works well, what failure cases are present, suggestions to improve the performance, etc.
    2. several folders with ground truth bounding boxes drawn as well as bounding boxes drawn from predictions of the model on validation dataset, for the sake of visualisation. 
    3. `object_det_evals.pdf` which contains some literature review that discusses some metrics that go beyond standard object detection evaluation metrics such as AP, as some of the standard evaluation metrics can prove to be not so useful when evaluating models for the task of autonomous driving.