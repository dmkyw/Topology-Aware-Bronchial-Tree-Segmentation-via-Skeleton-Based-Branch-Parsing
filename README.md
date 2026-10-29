# Topology-Aware-Bronchial-Tree-Segmentation-via-Skeleton-Based-Branch-Parsing
Topology-aware bronchial tree segmentation from CT. Airway masks are extracted using TotalSegmentator, then skeletonized with Kimimaro. Branch points are detected and parsed using rule-based anatomical constraints to produce consistent airway labeling and multi-segment outputs for 3D Slicer.

# 🫁 Bronchial Branch Labeler

Automatically parse an unsegmented bronchial tree into anatomically meaningful airway branches using skeleton-based topology analysis.

---

## 📌 Overview

This project provides a lightweight, **non-deep-learning** pipeline to convert a binary bronchial tree into structured anatomical segments (e.g., Trachea, RMB, RUL, B1–B10).

Unlike neural network approaches, this method is:

- Deterministic  
- Topology-aware  
- Interpretable (based on graph structure and geometry)  

It is particularly useful for:

- Airway anatomy analysis  
- Preprocessing for radiology AI pipelines  
- Generating labeled airway maps from CT-derived segmentations  

---

## 🧠 Pipeline
CT scan (.nii.gz)
↓
TotalSegmentator
↓
Extract bronchial tree (binary mask / Segment_1)
↓
Skeletonization (kimimaro)
↓
Graph-based branch tracing
↓
Anatomical classification (B1–B10)
↓
Export:

.seg.nrrd (3D Slicer compatible)
.txt (segment names)


---

## 🖼️ Demo

### Bronchial branch parsing

Below shows how the algorithm converts an **unsegmented bronchial tree** into labeled anatomical branches:

![Bronchial Branch Parsing](DemoImg/Bronchial%20Branch%20Parsing.jpg)

---

### Clinical visualization benefit

The colorized airway branches significantly improve interpretability when reviewing CT in different views:

- Axial  
- Coronal  
- Sagittal  

This allows faster identification of bronchial segments (e.g., B1–B10) during image navigation.

![Bronchial Branch Parsing 2](DemoImg/Bronchial%20Branch%20Parsing%202.jpg)

🎥 Video demonstration:  
https://youtu.be/xXrjiTo91TU

---

## ⚙️ Installation

```bash
pip install numpy scipy scikit-learn nibabel pynrrd kimimaro
```
## 🚀 Usage
```bash
python bronchial_branch_labeler.py ^
  --input "D:/LyNoS_dataset/Benchmark/Pat2/lung_vessels_segmentation/lung_vessels_segmentation_bronchial_ori.seg.nrrd" ^
  --ct "D:/LyNoS_dataset/Benchmark/Pat2/pat2_data.nii.gz" ^
  --output "D:/LyNoS_dataset/Benchmark/Pat2/lung_vessels_segmentation/bronchial_BRANCHES.seg.nrrd"
```

## Arguments
Argument	Description
--input	Binary bronchial tree segmentation (.seg.nrrd)
--ct	Original CT volume (.nii.gz) for spatial reference
--output	Output segmented airway file (.seg.nrrd)
--ap-sign (optional)	Set to -1 if anterior–posterior direction is flipped

## 📤 Output
1. Segmentation file
*.seg.nrrd
Compatible with 3D Slicer
Each airway branch is a separate segment layer
2. Segment name file
*_segment_names.txt
Contains ordered anatomical labels
3. Console output
```bash
=== FINAL SEGMENTS ===
 1. Trachea (bid=0)
 2. LMB (bid=1)
 3. RMB (bid=2)
 4. RUL (bid=3)
 ...
28. Lt_B10 (bid=38)
```

## 🔬 Method Highlights
Skeletonization via kimimaro
Graph traversal (BFS, shortest path)
Physical RAS orientation (superior, right, anterior)
Branch classification using geometric heuristics
Cranial direction → B1
Anterior/Posterior → B2 / B3
Spatial clustering for distal branches
Topology-preserving labeling

No deep learning model is used.

## ⚠️ Notes
Input bronchial tree should be reasonably clean (e.g., from TotalSegmentator)
Extremely noisy segmentations may affect branch detection
Left B7 is not defined (consistent with anatomical convention)

### Input and geometry requirements

- `--input` must be one 3D scalar NRRD labelmap. The selected airway voxels are those equal to `--airway-value` (default `1`). Multi-layer/4D Slicer segmentations must be exported as a single 3D mask first.
- The mask must contain observed branching. A single tube without graph junctions produces a structured `NO_BRANCH_POINTS` failure rather than fabricated branches.
- The NRRD must include an explicit RAS/LPS-family space, origin, and three orthogonal voxel directions. Sheared or unknown-space geometry is rejected.
- Voxel spacing and axis orientation are converted to physical RAS millimetres for skeleton analysis and voxel assignment.
- If `--ct` is supplied, it must be a 3D NIfTI with the same shape and voxel-to-world geometry as the input mask. The tool does not resample or copy a mismatched affine.
- Without `--ct`, output geometry is converted from the input NRRD and retained in RAS millimetres.

This repository labels an existing airway mask; it does not run TotalSegmentator. The output is a research result and still requires visual review in 3D Slicer. Geometry checks do not establish anatomical labeling accuracy or clinical suitability.

### Parsing modes and regression fix

- `ANATOMICAL_CANDIDATES`: the observed tree satisfies the naming heuristics. Anatomical names remain unvalidated candidates.
- `TOPOLOGY_ONLY`: the tree is partial, disconnected, cyclic, or otherwise insufficient for the naming rules. Observed skeleton paths receive neutral names such as `Topology_Branch_001`. Console diagnostics explain why anatomical naming was unresolved.
- Add `--strict-anatomy` to reject unresolved anatomical naming instead of using neutral paths. Unknown internal errors still fail; they are not converted into neutral results.

The parser now follows the most superior endpoint to the first junction and excludes that proximal tracheal limb when selecting the main bronchi. Immediate junctions are retained, and anatomical directions use RAS instead of unstable PCA signs. A skeleton without branching is retried at a 10 mm TEASAR constant after the initial 50 mm attempt. Smaller constants that produced false terminal spurs in a straight-tube regression are not used.

No mode fills missing airways, connects gaps, or changes the source mask. Disconnected components are retained for neutral parsing. Branch voxel assignment uses physical distances and preserves the selected foreground union.

### Reproducing the Windows environment

The tested environment is Windows with Python **3.12.10**. `requirements-lock.txt` records the complete installed package set, including Kimimaro **5.8.5**. In a separate Python 3.12 environment:

```powershell
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The five graph regressions cover proximal-limb exclusion, immediate junctions, explicit neutral fallback, fixed RAS directions, and tracheal root traversal. These tests establish implementation behavior, not anatomical accuracy.

## 📄 License

MIT License

## 🙌 Acknowledgement
TotalSegmentator for whole-body CT segmentation
kimimaro for skeletonization

## 💡 Future Work
Support for airway variants
Integration with radiology AI pipelines
Quantitative airway analysis (diameter, angle, topology metrics)
