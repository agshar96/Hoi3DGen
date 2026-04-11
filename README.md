# Hoi3DGen: Generating High-Quality Human-Object-Interactions in 3D

<p align="center">
  <a href="https://virtualhumans.mpi-inf.mpg.de/hoi3dgen/"><b>[🌐 Project Page]</b></a>
  <a href="https://arxiv.org/abs/2603.12126"><b>[📄 Paper]</b></a>
</p>

This is the official repository for **Hoi3DGen.**

---

## 🚀 News
- **[March 2026]** Our paper has been submitted to arXiv!
- **[Feb 2026]** Hoi3DGen has been accepted to CVPR Findings Track!
- **[Coming Soon]** We will release the full code, pre-trained models, and the annotated ProciGen Dataset.
- **⭐ Star us** to get notified when the code is released!


## 📦 Content to be Released
- [ ] **Core Code**: Training and inference pipeline.
- [ ] **Model Weights**: Pre-trained checkpoints for Hoi3DGen.
- [ ] **Annotated Dataset**: Complete list of text annotations for human object interactions from ProciGen dataset.

## 📄 Abstract
Modeling and generating 3D human–object interactions from text is crucial for applications in AR, XR, and gaming. Existing approaches often rely on score distillation from text-to-image models, but their results suffer from the Janus problem and do not follow text prompts faithfully due to the scarcity of high-quality interaction data. We introduce **Hoi3DGen**, a framework that generates high-quality textured meshes of human-object interaction that follow the input interaction descriptions precisely. 
We first curate realistic and high-quality interaction data leveraging multimodal large language models, and then create a full text-to-3D pipeline, which achieves orders-of-magnitude improvements in interaction fidelity. Our method surpasses baselines by 4-15x in text consistency and 3-7x in 3D model quality, exhibiting strong generalization to diverse categories and interaction types, while maintaining high-quality 3D generation.

---

## ✍️ Citation
If you find our work or code useful for your research, please consider citing:

```bibtex
@inproceedings{Sharma_and_xie2026Hoi3DGen,
      title={Hoi3DGen: Generating High-Quality Human-Object-Interactions in 3D},
      author={Agniv Sharma and Xianghui Xie and Tom Fischer and Eddy Ilg and Gerard Pons-Moll},
      booktitle = {CVPR findings},
        month = {June},
        year = {2026},
}
```

---
