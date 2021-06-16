# Video Summarization: Final Computer Vision 18TN Project
## Using [DSNet](https://ieeexplore.ieee.org/document/9275314) & [MSVA](https://arxiv.org/pdf/2104.11530.pdf)
*Original project available [here](https://github.com/li-plus/DSNet)*

## Bắt đầu
*Do chưa thật sự chạy được code ở bài báo MSVA nên ở đây ta chỉ huấn luyện, chạy mô hình DSNet.*

Hướng dẫn chạy thử trên Google Colab.
### Tạo mới 1 file .ipynb, cài đặt vị trí lưu trong google drive
```sh
from google.colab import drive
drive.mount('/content/drive')
% cd '/content/drive/My Drive/'
```

Clone project về
```sh
! git clone https://github.com/idkhoa1605/DSNet.git
```

Cài đặt các thư viện cần thiết
```sh
! pip install -r requirements.txt
```

### Chuẩn bị tập dữ liệu

Tải các tập dữ liệu đã được xử lí vào thư mục `datasets/`, bao gồm [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), và [YouTube](https://sites.google.com/site/vsummsite/download).

```sh
!mkdir -p datasets/ 
% cd datasets/
! wget https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip
! unzip dsnet_datasets.zip
```

Nếu link tập dữ liệu trên không khả dụng

+ (Baidu Cloud) Link: https://pan.baidu.com/s/1LUK2aZzLvgNwbK07BUAQRQ Extraction Code: x09b
+ (Google Drive) https://drive.google.com/file/d/11ulsvk1MZI7iDqymw9cfL7csAYS0cDYH/view?usp=sharing

Thư mục `datasets/` sẽ có dạng như sau:

```
DSNet
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```

### Mô hình đã được huấn luyện

Mô hình đã được tác giả huấn luyện sẵn. Bạn có thể sử dụng hoặc bỏ qua.

```sh
! mkdir -p models 
% cd models
# anchor-based model
! wget https://www.dropbox.com/s/0jwn4c1ccjjysrz/pretrain_ab_basic.zip
! unzip pretrain_ab_basic.zip
# anchor-free model
! wget https://www.dropbox.com/s/2hjngmb0f97nxj0/pretrain_af_basic.zip
! unzip pretrain_af_basic.zip
```

Để đánh giá mô hình đã huấn luyện trước:

```sh
# evaluate anchor-based model
! python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
# evaluate anchor-free model
! python evaluate.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

Nếu không có vấn đề xảy ra thì kết quả sẽ xấp xỉ:

|              | TVSum | SumMe |
| ------------ | ----- | ----- |
| Anchor-based | 62.05 | 50.19 |
| Anchor-free  | 61.86 | 51.18 |

## Huấn luyện mô hình

### Anchor-based

Để huấn luyện mô hình anchor-based trên tập dữ liệu TVSum:

```sh
! python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml
```

Huấn luyện với các phương thức khác như augmentation, transfer, LSTM...

```sh
! python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml
! python train.py anchor-based --model-dir ../models/ab_summe_aug/ --splits ../splits/summe_aug.yml
! python train.py anchor-based --model-dir ../models/ab_tvsum_trans/ --splits ../splits/tvsum_trans.yml
! python train.py anchor-based --model-dir ../models/ab_summe_trans/ --splits ../splits/summe_trans.yml
```

To train with LSTM, Bi-LSTM or GCN feature extractor, specify the `--base-model` argument as `lstm`, `bilstm`, or `gcn`. For example,

```sh
! python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm
```

### Anchor-free

Tương tự như anchor based

```sh
! python train.py anchor-free --model-dir ../models/af_basic --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```

Lưu ý: NMS threshold khuyến nghị là 0.4

## Đánh giá mô hình sau khi huấn luyện

Đánh giá mô hình anchor based
```sh
! python evaluate.py anchor-based --model-dir ../models/ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
```

Đánh giá mô hình anchor free

```sh
! python evaluate.py anchor-free --model-dir ../models/af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml --nms-thresh 0.4
```


## Huấn luyện với tập dữ liệu tùy ý

### Training & Validation

Để hiểu rõ hơn về cách tạo tập dữ liệu huấn luyện riêng, xem video mẫu ở thư mục `custom_data/videos` và nhãn mẫu ở `custom_data/labels`

Trong đó, thư mục `custom_data/videos` chứa các video cần huấn luyện, thư mục `custom_data/labels` gồm các file .json với nội dung là vector (M x N); M là số người đánh giá (tùy ý > 1), N là số frames của video (mang giá trị 0: không xuất hiện, 1: xuất hiện) 

#### Nếu chưa biết số frames của video có thể dùng code sau:
```sh
import cv2
cap = cv2.VideoCapture("sample.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )
```

Tạo tập dữ liệu dựa vào các video trong `custom_data/videos`

```sh
! python make_dataset.py --video-dir ../custom_data/videos --label-dir ../custom_data/labels \
  --save-path ../custom_data/custom_dataset.h5 --sample-rate 15
```

Sau đó tách ra vài video để làm tập test:

```sh
! python make_split.py --dataset ../custom_data/custom_dataset.h5 \
  --train-ratio 0.67 --save-path ../custom_data/custom.yml
```

Huấn luyện mô hình dựa trên video đã chọn

```sh
python train.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
python evaluate.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
```

### Dự đoán với video tùy ý

Để tóm tắt video, ta chạy code sau:

```sh
python infer.py anchor-based --ckpt-path ../models/custom/checkpoint/custom.yml.0.pt \
  --source ../custom_data/videos/EE-bNr36nyA.mp4 --save-path ./output.mp4
```

 Với tham số sau:
 - ```--ckpt-path```: Đường dẫn tới checkpoint tùy ý, vd: ```../models/ab_basic/checkpoint/tvsum.yml.0.pt```
 - ```--source```: Đường dẫn tới video cần tóm tắt, vd: ```./sample.mp4```
 - ```--save-path```: Đường dẫn tới vị trí lưu kết quả, vd: ```../output/sum_1.mp4```

## Acknowledgments

We gratefully thank the below open-source repo, which greatly boost our research.

+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.

## Citation

If you find our codes or paper helpful, please consider citing.

```
@article{zhu2020dsnet,
  title={DSNet: A Flexible Detect-to-Summarize Network for Video Summarization},
  author={Zhu, Wencheng and Lu, Jiwen and Li, Jiahao and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={948--962},
  year={2020}
}
```
