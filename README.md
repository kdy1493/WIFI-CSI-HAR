# WiFi CSI Processing

A Python project for collecting, processing, and analyzing WiFi CSI (Channel State Information) data.

## Project Overview

This project provides a complete pipeline for WiFi CSI data:
- Convert raw CSI data to CSV format
- Data preprocessing and filtering
- Statistical analysis and visualization
- Apply dimensionality reduction techniques like PCA

## Key Features

### Data Processing
- **Raw CSI to CSV**: Convert raw CSI data to CSV format
- **Preprocessing**: Noise removal and signal filtering
- **Data Merging**: Combine multiple CSV files into one

### Analysis and Visualization
- **CSI Data Visualization**: Visualize raw and processed CSI data
- **Signal Analysis**: Frequency domain analysis
- **PCA Analysis**: Pattern analysis through dimensionality reduction
- **Subcarrier Analysis**: Individual subcarrier characteristic analysis

## Installation

### Requirements
- Python 3.11 or higher
- uv (Python package manager) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd WIFI-CSI-PROCESSING
```

2. Create virtual environment and install dependencies:

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Inference 
```bash 
uv run scripts/CSI_Plot_Main.py
```


### Advanced Analysis

Use Jupyter notebooks for detailed analysis:

```bash
jupyter notebook nbs/
```

- `01_plotting_raw_csi.ipynb`: Raw CSI data exploration
- `04_signal_filtering.ipynb`: Signal filtering techniques
- `05_PCA.ipynb`: Principal component analysis

## Dependencies

Key libraries:
- `numpy`, `pandas`: Data processing
- `matplotlib`: Visualization
- `scikit-learn`: Machine learning and PCA
- `scipy`: Scientific computing
- `paho-mqtt`: MQTT communication (real-time data collection)

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## License

This project is distributed under the MIT License. See `LICENSE` file for more information.

## Contact

If you have any questions or suggestions about the project, please create an issue.
