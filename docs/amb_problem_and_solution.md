# AMB – Autonomous Minecraft Builder

## Overview

**AMB (Autonomous Minecraft Builder)** is a research-driven system that learns how to **build Minecraft structures autonomously**, one action at a time, instead of predicting entire structures in a single shot.

The core objective is not image-like generation, but **procedural construction**:

> deciding _what block to place_, _where to place it_, and _when to stop_.

AMB treats Minecraft building as a **sequential decision-making problem**, closer to planning than static prediction.

---

## Original Goal

Train a model that can:

- Understand real Minecraft structures
- Learn construction logic
- Build complex structures step-by-step from an empty world

---

## The Real Problem We Hit

### ❌ Data scarcity

- Only ~2000 real schematics available
- Insufficient for voxel-level supervised learning
- Scraping popular sites is restricted or unethical

### ❌ Synthetic data failed

- Synthetic builds lacked global structure
- Model learned local patterns only
- Failed on real-world complexity

### ❌ Fine-tuning didn’t help

- Pretraining learned the wrong abstractions
- Real data could not “correct” synthetic bias
- More epochs = more wasted electricity

At this point, **data became the problem**, not the model.

---

## Core Insight (The Turning Point)

**Schematics are STATIC.  
Building is PROCEDURAL.**

Therefore:

- ❌ Predicting full structures does not work
- ✅ Predicting **one build action per step** does

The model was failing because it was trained on the wrong task.

---

## Correct Reformulation of the Problem

Instead of learning:

> “What does the final structure look like?”

AMB learns:

> “Given the current world state, what is the next valid build action?”

This reframing:

- Removes ambiguity
- Amplifies limited data
- Aligns training with how building actually works

---

## The Solution: Procedural Build Pipeline

### High-Level Pipeline

1. Load real schematics
2. Simplify noisy real-world data
3. Normalize coordinates
4. Segment structures into build phases
5. Generate valid build sequences
6. Convert sequences into step-wise training samples
7. Train using curriculum learning
8. Verify by overfitting a single complex build

---

## Key Techniques That Solve the Data Problem

### 1. Data Amplification (Without Scraping)

Each schematic is converted into:

- Multiple valid build sequences
- Thousands of `(state → next action)` samples

Result:

- 2000 schematics → hundreds of thousands or millions of training steps

---

### 2. Structure Simplification

Real builds are noisy and decorative.

We:

- Remove decor (torches, banners, plants, etc.)
- Collapse block variants
- Focus on structural blocks first

This makes real data **learnable**.

---

### 3. Build Phases (Hierarchy)

Each block is assigned a phase:

- FOUNDATION
- WALL
- ROOF
- WINDOW
- DETAIL

The model is conditioned on:

- current phase
- global progress

This injects **planning and hierarchy**.

---

### 4. Ordered Build Sequences

Static schematics are converted into **temporal trajectories**:

- bottom → top
- inside → outside
- no floating blocks
- phase-by-phase construction

Time is restored to the data.

---

### 5. Explicit STOP Token

The model predicts:

- `(dx, dy, dz, block_type)`
- or `<STOP>`

This prevents infinite hallucinated building.

---

### 6. Curriculum Training

Training complexity increases gradually:

1. Synthetic primitives
2. Simplified real structures
3. Structured real builds
4. Full real builds with detail

Mixing everything at once is forbidden.

---

## Mandatory Sanity Checks

Before scaling:

- [ ] Overfit **one** simplified real structure
- [ ] Rebuild it perfectly
- [ ] Stop correctly
- [ ] Handle different valid build orders

If any fail:

> Stop training. Fix representation or ordering.

---

## What AMB Is (And Is Not)

### AMB **is**:

- Procedural
- Constraint-aware
- Data-efficient
- A system, not just a neural net

### AMB **is not**:

- End-to-end magic
- A pure voxel prediction task
- Solvable by “more data + more epochs”

---

## Current Status

- Problem correctly identified
- Pipeline designed to fix data limitations
- Focus shifted from scraping to **structural intelligence**

AMB is now a **serious research problem**, not a dataset chase.
