**BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning (Images 100K)** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the automotive industry. 

The dataset consists of 10000 images with 205552 labeled objects belonging to 10 different classes including *car*, *pedestrian*, *truck*, and other: *bus*, *bicycle*, *rider*, *motorcycle*, *caravan*, *trailer*, and *train*.

Images in the BDD100K: Images 10K dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 2128 (21% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *train* (7000 images), *test* (2000 images), and *val* (1000 images). The dataset was released in 2020 by the UC Berkeley, USA, Cornell University, USA, UC San Diego, USA, and Element, Inc.

Here are the visualized examples for the classes:

[Dataset classes](https://github.com/dataset-ninja/bdd100k-10k/raw/main/visualizations/classes_preview.webm)
