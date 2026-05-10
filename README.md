# Sentinel

Sentinel is a computer vision project focused on traffic and safety monitoring, including helmet detection and speed violation workflows.

## Repository Structure

- `backend/` - Backend services and APIs.
- `models/` - Machine learning models and experiments.
  - `HelmetDetection/` - Helmet detection dataset, scripts, and environment files.
  - `SpeedDetection/` - Speed detection scripts, model weights, and violation outputs.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/gaurangpatil97/Sentinel2.git
   cd Sentinel2
   ```

2. Set up Python environments as needed:
   - `models/HelmetDetection/helmetvenv/`
   - `models/modelsvenv/`

3. Install dependencies for the module you want to run. For example:
   ```bash
   cd models/HelmetDetection
   pip install -r requirements.txt
   ```

4. Run scripts from their respective folders:
   - Helmet detection helpers: `models/HelmetDetection/checkDataset.py`, `downloadDataset.py`
   - Speed detection pipeline: `models/SpeedDetection/v1.py`

## Notes

- Model files and generated outputs (for example, violation reports) are stored under `models/SpeedDetection/`.
- Dataset metadata for helmet detection is available in `models/HelmetDetection/dataset/`.

## License

Add your preferred license details here.
