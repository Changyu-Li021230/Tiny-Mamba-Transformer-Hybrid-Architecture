# TinyMambaFormer: A Tiny Mamba-Transformer Hybrid for Efficient Human Activity Recognition (HAR)

**Project Author**: Changyu 
**Date**: June 2025  
**Keywords**: Human Activity Recognition, Mamba, Transformer, Edge AI, Multimodal Fusion, Temporal Modeling

---

##  Overview

**TinyMambaFormer** is a lightweight hybrid deep learning architecture designed for real-time **Human Activity Recognition (HAR)** on resource-constrained edge devices. It combines the localized encoding capabilities of CNNs, the mid-range temporal modeling power of Mamba (a state-space model), and the global reasoning strength of a compact Transformer.

This architecture enables **multi-modal**, **low-latency**, and **high-accuracy** HAR from signals such as IMU, WiFi CSI, and visual skeleton sequences.

---

##  Features

-  **Ultra-low latency** (< 40 ms) for real-time inference  
-  **Three-stage design**:
  - Local temporal encoding via lightweight depthwise separable convolutions
  - Mid-range temporal modeling via a frequency-decoupled Tiny Mamba block
  - Global context aggregation via a parameter-efficient Transformer
-  **Multimodal fusion support** (e.g., IMU + CSI + skeleton)
- <1MB model size – deployable on microcontrollers and mobile devices  
-  Evaluated on UCI-HAR, PAMAP2, WiAR, and Opportunity datasets

---

##  Architecture

