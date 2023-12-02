This is the **Images 10K** part of the **BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning**, which is the largest driving video dataset with 100K videos and 10 tasks, providing a comprehensive evaluation platform for image recognition algorithms in autonomous driving. The dataset boasts geographic, environmental, and weather diversity, enhancing the robustness of trained models. Through BDD100K, the authors establish a benchmark for heterogeneous multitask learning, demonstrating the need for specialized training strategies for existing models to handle such diverse tasks, thereby opening avenues for future research in this domain.

Currently, the following datasets are presented on a DatasetNinja platform:

- **BDD100K: Images 100K** ([available on DatasetNinja](https://datasetninja.com/bdd100k))
- **BDD100K: Images 10K** (current)

<img src="https://github.com/dataset-ninja/bdd100k/assets/78355358/fd71c0e2-4c71-4277-9e35-84886602a30d" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">The dataset includes a diverse set of driving videos under various weather conditions, time, and scene types. The dataset also comes with a rich set of annotations: scene tagging, object bounding box, lane marking, drivable area, full-framesemantic and instance segmentation, multiple object tracking, and multiple object tracking with segmentation.</span>

The authors construct BDD100K as a diverse and large-scale dataset of visual driving scenes. This dataset overcomes limitations by collecting over 100K diverse video clips, covering various driving scenarios, and capturing a broad range of appearance variations and pose configurations. The benchmarks encompass ten tasks, including image tagging, lane detection, drivable area segmentation, road object detection, semantic segmentation, instance segmentation, multi-object detection tracking, multi-object segmentation tracking, domain adaptation, and imitation learning. These diverse tasks enable the study of heterogeneous multitask learning, and the authors conduct extensive evaluations of existing algorithms on the new benchmarks, shedding light on the challenges of designing a single model for multiple tasks.

<img src="https://github.com/dataset-ninja/bdd100k/assets/78355358/087b0555-e1f8-480d-b171-93eed1c6d3bf" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Geographical distribution of the data sources. Each dot represents the starting location of every video clip. The videos are from
many cities and regions in the populous areas in the US.</span>

<img src="https://github.com/dataset-ninja/bdd100k/assets/78355358/d4a72cab-80cd-44b1-8f66-b3e342b2ba5b" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Instance statistics of our object categories. (a) Number of instances of each category, which follows a long-tail distribution. (b) Roughly half of the instances are occluded. \(c\) About 7% of the instances are truncated.</span>

To provide a large-scale diverse driving video dataset, the authors utilize a crowdsourcing approach with contributions from tens of thousands of drivers, supported by Nexar. BDD100K includes over 100K driving videos, each 40 seconds long, collected from more than 50K rides across locations like New York and the San Francisco Bay Area, offering diverse scene types and weather conditions. The dataset is split into training, validation, and testing sets, with annotations for image tasks at the 10th second in each video and the entire sequences used for tracking tasks.

## Tasks

The Image tagging involves annotations for weather conditions, scene types, and times of day. Object detection includes bounding box annotations for 10 categories, while lane marking involves labeling with eight main categories, continuity, and direction attributes. Drivable area detection distinguishes between directly and alternatively drivable areas. Semantic instance segmentation provides pixel-level annotations for 40 object classes in images randomly sampled from the dataset.

The authors present their dataset as an essential resource for advancing research in street-scene understanding, offering unparalleled diversity and complexity for evaluating algorithms in the domain of autonomous driving.
