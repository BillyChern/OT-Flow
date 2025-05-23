# OT-flow Project

This project implements a generative model pipeline combining a Variational Autoencoder (VAE),
an Optimal Transport (OT) map, and a Flow Matching model.

## Project Structure

- `dataset_loader.py`: Loads and preprocesses the dataset.
- `vae_model.py`: Defines the VAE architecture.
- `train_vae.py`: Script for training the VAE.
- `ot_model.py`: Defines the OT map model.
- `train_ot.py`: Script for training the OT map.
- `flow_matching_model.py`: Defines the Flow Matching model.
- `train_flow_matching.py`: Script for training the Flow Matching model.
- `generate.py`: Script to generate samples using the full pipeline.
- `configs/`: Directory for configuration files.
- `utils.py`: Utility functions.
- `requirements.txt`: Python dependencies.

## TODO
- Implement VAE
- Implement OT Map
- Implement Flow Matching
- Implement training and generation pipelines 