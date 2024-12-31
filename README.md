# RICH_x_ICSU

## DATA

### TOOL

```bash
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("yolov11-midrp").project("hoi-gmd2i")
version = project.version(4)
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

### TRAIN
- TRAIN.py

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

### CODE VERSION4.0
- ICSUv4.py

### PUBLICATION
- [RICHDRAFTv2.5.pdf](https://www.overleaf.com/project/676251175b52cb44758793fb)

### SHOWTIME
<a href="https://uflorida-my.sharepoint.com/:v:/g/personal/xulinfeng_ufl_edu/EXCJN0Z2U2FMrkVhWOUK7gIBUbDMBDlSAA6CUmPX7Kaq-A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=Z5s1h8">
  <img src="https://github.com/user-attachments/assets/79a8c0eb-3c47-43e0-8cb3-8e76d581763f" alt="Demo Image" style="width:1000px;">
</a>

