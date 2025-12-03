# Continual Learning with CLIP + LoRA on CIFAR-10

This repository contains a minimal research project implementing a **continual learning benchmark** using a pre-trained **CLIP vision encoder**, comparing:

- **Full fine-tuning**
- **LoRA (Low-Rank Adaptation)**
- **Catastrophic forgetting** on Task A after learning Task B

The objective is to reproduce the classical forgetting effect and explore how **parameter-efficient updates** can mitigate it in a lightweight and reproducible setup.

---

## Features

- **Sequential training** on CIFAR-10  
  - Task A: classes **0–4**  
  - Task B: classes **5–9**
- **CLIP + HuggingFace + PEFT integration**
- **Side-by-side comparison** of full fine-tuning vs. LoRA
- **Automatic visualization** of forgetting using comparison graphs
- **HPC-ready workflow**, including an optional Slurm job script

---

## Folder Structure


