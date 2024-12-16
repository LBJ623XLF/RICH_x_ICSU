# RICH_x_ICSU

## DATA

### TOOL

```bash
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("yolov11-midrp").project("hoi-gmd2i")
version = project.version(3)
dataset = version.download("yolov8")
```

### HUMAN

```bash
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("test-vongh").project("awkward-posture-of-human")
version = project.version(3)
dataset = version.download("yolov8")
```
## MODEL

### TOOL
- best.pt

### HUMAN
- pose.pt

## INDUSTRIALIZED CONSTRUACTION SCENE UNDERSTANDING

### CODE VERSION3.0
- ICSUv3.py

### PUBLICATION
- RICHDRAFT.pdf

### SHOW
![IMG_0132](https://github.com/user-attachments/assets/cebceb9d-b149-4afd-8a52-43fc8d36b7d9 |width="960")
