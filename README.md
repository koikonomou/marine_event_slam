## Installation

Requires [Poetry](https://python-poetry.org/).

```bash
git clone <repo-url>
cd marine_event_slam
poetry install
```

## Dataset

Tested on the [Maritime Visual Tracking Dataset (MVTD)](https://github.com/chenzx/MVTD).

Expected folder structure:

```
~/codes/datasets/Maritime_Visual_Tracking_Dataset_MVTD/
    train/
        119-USV/
            00000001.jpg
            00000002.jpg
            ...
            groundtruth.txt
```

## Usage

```bash
poetry run python marine_event_slam.py --frame /path/to/the/folder 
```
