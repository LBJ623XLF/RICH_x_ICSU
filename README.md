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

### CONFIG
- config.txt

### VIDEO
[![Demo Video](https://img.youtube.com/vi/blKOrb_HNVc/0.jpg)](https://www.youtube.com/watch?v=blKOrb_HNVc)
[![Demo Video](https://img.youtube.com/vi/HnkFMYtl7_g/0.jpg)](https://www.youtube.com/watch?v=HnkFMYtl7_g)

### CODE VERSION3.0
- ICSUv4.py

### PUBLICATION
- RICHDRAFT.pdf

### SHOW
![IMG_0132](https://github.com/user-attachments/assets/31641e68-23e0-4f8a-8fcf-e46669660d01)
https://uflorida-my.sharepoint.com/:v:/g/personal/xulinfeng_ufl_edu/ERlkpXNeRFBLsz0c_yeq2B8BWgNTkdiSj3zLnxLxeNo08Q?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=GyNJHE
