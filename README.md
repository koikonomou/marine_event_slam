## Installation

Requires [Poetry](https://python-poetry.org/).

```bash
git clone <repo-url>
cd marine_event_slam
poetry install
```

## Dataset

Tested on the [Maritime Visual Tracking Dataset (MVTD)](https://github.com/AhsanBaidar/MVTD).

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
poetry run python marine_event_detector.py --frame /path/to/the/folder 
```
## Stabilizer

```bash
python3 stabilizer.py -i  ../on-board-frames -o ../out3 -d ../debug3 -w 5 -b -sy 0.3 -dm 1.3
```
