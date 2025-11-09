<div align='center'>

<h2><a href="https://link.springer.com/chapter/10.1007/978-3-032-05169-1_18">EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</a></h2>

[Shanshan Song](https://scholar.google.com.hk/citations?user=EoNWyTcAAAAJ)<sup>1</sup>, [Yi Qin](https://scholar.google.com.hk/citations?user=oIcu4mgAAAAJ)<sup>1</sup>, [Honglong Yang](https://scholar.google.com/citations?user=3BPUjoQAAAAJ)<sup>1</sup>, Taoran Huang<sup>2</sup>, Hongwen Fei<sup>2</sup>, [Xiaomeng Li](https://scholar.google.com/citations?user=uVTzPpoAAAAJ)<sup>1</sup>
 
<sup>1</sup>Hong Kong University of Science and Technology (HKUST) 

<sup>2</sup>Guangdong Provincial People’s Hospital


</div>


## 🤖 Architecture

<p align="center">
   <img src="docs/fig1.jpg" alt="overview" width="800" />
</p>


## 🔨 Installation

Clone this repository and install the required packages:

```shell
git clone https://github.com/xmed-lab/EchoViewCLIP.git
cd EchoViewCLIP

conda create -n echoviewclip python=3.10
conda activate echoviewclip
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
pip install -r requirements.txt
cd apex/
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```


## 🍹 Preparation

### Dataset

In preparing the datasets, we adopted the same organizational format as [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP/blob/main/docs/DATASETS.md), structuring the data as follows:

-  **Standard Folder**. Put all videos in the `videos` folder, and prepare the annotation files as `train.txt` and `val.txt`. Please make sure the folder looks like this:
    ```Shell
    $ ls /PATH/TO/videos | head -n 2
    a.mp4
    b.mp4

    $ head -n 2 /PATH/TO/train.txt
    a.mp4 0
    b.mp4 2

    $ head -n 2 /PATH/TO/val.txt
    c.mp4 1
    d.mp4 2
    ``` 

Our 38 view labels can be found in labels/ultracls.csv.


## 🍻 Quick Start for Training & Evaluation
After all the above preparation steps, you can train EchoViewCLIP with the following command: 
```shell
# Stage1 training: Standard view classification
bash train_stage1.sh

# Stage2 training: Negation Semantic-Enhanced OOD Detector
bash train_stage2.sh

# Stage3 training: Quality Control
bash train_qc.sh

```
For evaluation, add --only_test to the command script, adjust the VAL_FILE path in the configuration .yaml file, and include --resume followed by the path to the model weights.

## 💙 Acknowledgement

EchoViewCLIP is built upon the awesome [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) and [CLIPN](https://github.com/xmed-lab/CLIPN).

## 📄 Citation

Paper link: [EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf).

If you use this work in your research, please cite:

```bibtex
@inproceedings{song2025echoviewclip,
  title={EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography},
  author={Song, Shanshan and Qin, Yi and Yang, Honglong and Huang, Taoran and Fei, Hongwen and Li, Xiaomeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={181--191},
  year={2025},
  organization={Springer}
}
```

## 📧 Contact

For questions and issues, please use the GitHub issue tracker or contact [ssongan@connect.ust.hk]. 