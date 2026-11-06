# Brain MRI Tumor Segmentation using U-Net

A mini project that segments possible tumor regions in brain MRI slices using a U-Net model built with PyTorch. The web interface is built with Streamlit.

> **Important:** This is an educational/research prototype, not a medical diagnosis system.

## Project files

```text
tumor-segmentation-streamlit/
├── app.py
├── model.py
├── train_model.py
├── requirements.txt
├── .gitignore
├── README.md
└── artifacts/
    ├── unet_best.pt
    └── metadata.json
```

The `artifacts/` folder is created after model training. It is intentionally excluded from Git by `.gitignore` because model checkpoints can be large.

## 1. Create a virtual environment (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build suitable for your CUDA version using the official PyTorch installation selector. CPU training is supported but may be very slow.

## 3. Train the model

```powershell
python train_model.py
```

The training script downloads the public LGG MRI Segmentation dataset through KaggleHub, creates a patient-wise 70/15/15 train/validation/test split, and trains the U-Net model.

After training, these files should exist:

```text
artifacts/
├── unet_best.pt
└── metadata.json
```

Training can take a long time and requires substantial RAM. It is recommended to train in Google Colab with a GPU, then copy the generated `artifacts/` folder into this project.

## 4. Run Streamlit

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Upload a brain MRI slice (PNG, JPG, or TIFF). The application displays the processed input, predicted binary mask, overlay, and estimated predicted area. You can download the mask as a PNG.

## 5. Push the source code to GitHub

Create a new empty repository on GitHub. In the project folder, run:

```powershell
git init
git add .
git commit -m "Add U-Net MRI tumor segmentation Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL.

### Important about model weights

The `.gitignore` excludes `artifacts/`. This avoids accidentally pushing a large model checkpoint. To deploy the app on Streamlit Community Cloud, the app must be able to access the trained model files. If the checkpoint is small enough for your repository and GitHub limits, you can remove `artifacts/` from `.gitignore` and commit it. For larger checkpoints, use Git LFS or external model storage and download the checkpoint during deployment.

## 6. Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open https://share.streamlit.io/
3. Sign in with GitHub.
4. Select **Create app** and choose your repository.
5. Set the main file path to `app.py`.
6. Deploy.

The trained checkpoint and metadata must be made available to the deployed app. Streamlit Community Cloud is not intended for training this model; train it separately and deploy the saved checkpoint.

## Methodology retained from the original project

- Dataset: LGG MRI Segmentation (`mateuszbuda/lgg-mri-segmentation`)
- Image size: 128 × 128
- Patient-wise split: 70% training, 15% validation, 15% test
- U-Net encoder-decoder with skip connections
- Loss: Binary Cross-Entropy + Dice loss
- Optimizer: AdamW
- Data augmentation: horizontal flip, rotation, zoom, and brightness adjustment
