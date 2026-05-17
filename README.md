<div align="center">

# AMB - Autonomous Minecraft Builder

### Research-oriented ML project for step-by-step Minecraft structure generation

<p>
  <img src="https://img.shields.io/badge/Python-111827?style=for-the-badge" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-111827?style=for-the-badge" alt="PyTorch" />
  <img src="https://img.shields.io/badge/ML-111827?style=for-the-badge" alt="ML" />
  <img src="https://img.shields.io/badge/Minecraft-111827?style=for-the-badge" alt="Minecraft" />
  <img src="https://img.shields.io/badge/Datasets-111827?style=for-the-badge" alt="Datasets" />
  <img src="https://img.shields.io/badge/Research-111827?style=for-the-badge" alt="Research" />
</p>
<p>
  <a href="https://github.com/VortexDevX/AMB"><img src="https://img.shields.io/badge/GitHub%20Repo-111827?style=for-the-badge" alt="GitHub Repo" /></a>
</p>

</div>

---

## Overview

AMB explores how a model can learn Minecraft building as a sequence of actions rather than a one-shot structure prediction. The project includes dataset tooling, model components, training scripts, inference code, and experiments.

<table>
<tr>
<td width="25%"><strong>Status</strong></td>
<td>Experimental ML repository</td>
</tr>
<tr>
<td><strong>Stack</strong></td>
<td>Python, PyTorch, NumPy, custom datasets, training scripts, notebooks</td>
</tr>
<tr>
<td><strong>Built for</strong></td>
<td>ML builders exploring sequential structure generation in Minecraft</td>
</tr>
</table>

## Highlights

- Sequential building framed as a machine learning problem
- Dataset preparation and organization utilities
- Model components for state encoding, action heads, and builder transformers
- Training and inference modules kept behaviorally unchanged
- No checkpoint or dataset deletion unless clearly generated and ignored

## Feature Map

<table>
<tr>
<td width="50%" valign="top">

### Data Pipeline

Load, segment, simplify, organize, and prepare Minecraft structure data.

</td>
<td width="50%" valign="top">

### Model Code

State encoders, action heads, builder transformer, losses, and generators.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Training

Scripts and notebooks for experiments and curriculum-style training.

</td>
<td width="50%" valign="top">

### Research Notes

Docs and prompts capture problem framing and experiment direction.

</td>
</tr>
</table>

## Quick Start

```bash
cd amb
pip install -r requirements.txt
python -m pytest ../test
```

## Project Structure

- amb/ - package source
- amb/data/ - dataset and preprocessing tools
- amb/models/ - model architecture pieces
- amb/training/ - training code and losses
- docs/ - research notes
- test/ - tests and visualization helpers

## Notes

- ML architecture was not changed.
- Notebook content was not rewritten.
- Large checkpoints and datasets should stay ignored unless intentionally versioned.

---

<div align="center">

<strong>Clean docs. Clear setup. No fake screenshots.</strong>

</div>
