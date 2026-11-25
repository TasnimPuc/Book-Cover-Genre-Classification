# Book Cover Genre Classification

Predicting a book's genre (32 classes) from its **cover image alone**, using transfer learning with **ResNet-101** and **EfficientNet-B4** (PyTorch).

## Dataset

- **Used:** [Amazon Book Covers 207k – 32 Genres](https://www.kaggle.com/datasets/tasnimakters/amazon-book-covers-207k-32-genres) (Kaggle)
- **Modified from:** [uchidalab/book-dataset](https://github.com/uchidalab/book-dataset) — *Judging a Book by its Cover* (Iwana et al.)

`book32-listing_update1.csv` (included) holds the metadata for 207,570 books:

| Column | Description |
|---|---|
| `IMAGE URL` | Cover image file name |
| `TITLE` | Book title |
| `AUTHOR` | Author |
| `CATEGORY ID` | Genre label (0–31) |
| `CATEGORY` | Genre name |

Cover images are **not** included in this repo — download them from the Kaggle link above.

<details>
<summary>32 genre classes</summary>

| ID | Genre | ID | Genre |
|---|---|---|---|
| 0 | Arts & Photography | 16 | Medical Books |
| 1 | Biographies & Memoirs | 17 | Mystery, Thriller & Suspense |
| 2 | Business & Money | 18 | Parenting & Relationships |
| 3 | Calendars | 19 | Politics & Social Sciences |
| 4 | Children's Books | 20 | Reference |
| 5 | Comics & Graphic Novels | 21 | Religion & Spirituality |
| 6 | Computers & Technology | 22 | Romance |
| 7 | Cookbooks, Food & Wine | 23 | Science & Math |
| 8 | Crafts, Hobbies & Home | 24 | Science Fiction & Fantasy |
| 9 | Christian Books & Bibles | 25 | Self-Help |
| 10 | Engineering & Transportation | 26 | Sports & Outdoors |
| 11 | Health, Fitness & Dieting | 27 | Teen & Young Adult |
| 12 | History | 28 | Test Preparation |
| 13 | Humor & Entertainment | 29 | Travel |
| 14 | Law | 30 | Gay & Lesbian |
| 15 | Literature & Fiction | 31 | Education & Teaching |

</details>

## Method

- 207,552 covers matched to the CSV, stratified split (seed 42): **Train 103,776 / Val 72,643 / Test 31,133**
- Augmentation: `RandomResizedCrop(224)`, `RandomHorizontalFlip`; eval: `Resize(256)` → `CenterCrop(224)`; ImageNet normalization
- ImageNet-pretrained backbone, final layer replaced with a 32-class head, full fine-tuning
- `AdamW (lr=1e-3)`, `CrossEntropyLoss`, `ReduceLROnPlateau (factor 0.1, patience 3)`, mixed precision (AMP), batch 128, 15 epochs
- Best checkpoint saved by validation accuracy

## Results (test set, 32 classes)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| ResNet-101 | 35.1% | 0.316 | 0.296 | 0.297 |
| **EfficientNet-B4** | **38.4%** | 0.352 | 0.335 | 0.340 |

Random chance with 32 classes is ~3%. Covers alone are an ambiguous signal for genre, so most errors come from visually overlapping genres. Visually distinctive classes (Test Preparation, Travel, Romance) score highest.

> EfficientNet macro metrics are from the evaluation section of `efficient_net.ipynb`, which re-splits with `test_size=0.4` (41,511 images). The 38.4% figure is from the original 31,133-image test split.

## Repository structure

```
├── 2177-A_MLL_Project_Code Resnet.ipynb   # ResNet-101 training + evaluation
├── efficient_net.ipynb                    # EfficientNet-B4 training + evaluation
├── book32-listing_update1.csv             # Dataset metadata / labels
├── requirements.txt
└── README.md
```

Notebooks include: training curves, classification report, per-class accuracy, confusion matrix, macro-average ROC/AUC.

## Trained weights

`resnet_best.pth` (171 MB) and `efficientnet_best.pth` (71 MB) are not stored in git (GitHub's 100 MB file limit). Download them from the [Releases](../../releases) page.

```python
import torch, torch.nn as nn
from torchvision.models import efficientnet_b4

model = efficientnet_b4()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 32)
model.load_state_dict(torch.load("efficientnet_best.pth", map_location="cpu"))
model.eval()
```

## How to run

The notebooks were written for **Google Colab** (GPU).

1. Download the cover images from Kaggle, zip them as `book_images_small.zip`, and put it in Google Drive (`MyDrive/`).
2. Upload `book32-listing_update1.csv` to Drive as `MyDrive/book_list.csv` (or edit `csv_path` in the notebook).
3. Open a notebook in Colab, select a GPU runtime and run all cells.

Local: `pip install -r requirements.txt`, then update `img_dir` / `csv_path` and remove the Drive-mount cell.

## Acknowledgements

- B. K. Iwana, S. T. Raza Rizvi, S. Ahmed, A. Dengel, S. Uchida, *"Judging a Book by its Cover"*, arXiv:1610.09204, 2016 — [uchidalab/book-dataset](https://github.com/uchidalab/book-dataset)
- Kaggle dataset by [tasnimakters](https://www.kaggle.com/datasets/tasnimakters/amazon-book-covers-207k-32-genres)
