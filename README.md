# BCI Competition IV - Dataset 2a

This project uses EEG data from the [BCI Competition IV Dataset 2a](http://www.bbci.de/competition/iv/).

## 📁 Data Setup

**Note:** The dataset is **not included** in this repository due to its size (~420 MB zipped) and licensing.

### 🔗 Download Instructions

1. Go to the [official competition page](http://www.bbci.de/competition/iv/).
2. Locate **Dataset 2a** and download the following:
   - `A01T.gdf` to `A09T.gdf` (training files)
   - `A01E.gdf` to `A09E.gdf` (evaluation files)
3. Save the files to the following directory structure:
    data/ 
    └── raw/ 
        ├── A01T.gdf ├── A01E.gdf └── ...

### ⚠️ Automatic Setup Script

Run the script `scripts/download_data.py` to verify the presence of required files and optionally download metadata.

> Don't forget to run preprocessing scripts before training your models!

## 📚 About the Dataset

- 9 subjects
- 4-class motor imagery task
- Recorded with 22 EEG + 3 EOG channels
- Sample rate: 250 Hz

For full details, refer to the [technical documentation PDF](http://www.bbci.de/competition/iv/desc_2a.pdf).

