# DeNAT - Deep Neurite Analysis Tool

A machine-learning framework for precise automated neurite outgrowth measurement.

DeNAT is an accessible deep-learning-based platform that automatically measures neurite outgrowth after injury.

**🔗 Live App: [neuriteanalysis.netlify.app](https://neuriteanalysis.netlify.app/)**

## Demo

Watch the demo video: [DeNAT_Demo.MP4](DeNAT_Demo.MP4)

## Requirements

### Model
- TensorFlow 2.0+
- tqdm

### XRAI Saliency
- NumPy
- OpenCV
- Scikit-Image
- tqdm

### Concept Score
- NumPy
- Pandas
- Scikit-Learn
- SciPy
- tqdm

### Install All
```bash
pip install tensorflow numpy opencv-python scikit-image pandas scikit-learn scipy tqdm
```

## Usage

1. Open `index.html` in your browser
2. Upload your microscopy image
3. Draw a midline as your reference point
4. Select the analysis side (Left/Right)
5. Define the analysis area around your neurites
6. Click Auto-Detect to run the ML model
7. Review and refine results if needed
8. Export your data as CSV

## Folder Structure

```
├── model/           # Trained model files
├── interpret/       # XRAI saliency & interpretation scripts
├── Example data/    # Sample input images
├── Example_output/  # Sample results
└── index.html       # Main application interface
```

## Documentation

See [DeNAT Analysis Workflow.pdf](Example_output/DeNAT%20Analysis%20Workflow.pdf) for the complete visual guide with screenshots.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

© 2024 DeNAT. Free for academic and research use.
