import cv2
import torch
from IPython.display import clear_output
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models
from super_gradients.training import Trainer
#-- GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
  print(torch.cuda.get_device_name(0))

CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='potato_yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)

#커스텀 데이터셋 가져오기
from roboflow import Roboflow
rf = Roboflow(api_key="zvkzBMi7kSL5hhiyz46i")
project = rf.workspace("yoloprac-ljuxs").project("sweet-or-not")
version = project.version(3)
dataset = version.download("yolov5")

dataset_params = {
    'data_dir':'sweet-or-not-3',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['potato', 'sweetpotato']
}

from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
        # 'show_all_warnings': True
    },
    dataloader_params={
        'batch_size':10,
        'num_workers':5
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':10,
        'num_workers':5
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':10,
        'num_workers':5
    }
)

clear_output()

train_data.dataset.transforms
train_data.dataset.dataset_params['transforms'][1]
train_data.dataset.dataset_params['transforms'][1]['DetectionRandomAffine']['degrees'] = 10.42

model = models.get('yolo_nas_s',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )

train_params = {
    # 훈련 중 출력 억제
    'silent_mode': True,
    # 모델들의 평균을 사용하여 최종 모델 생성
    "average_best_models":True,
    "warmup_mode": "LinearEpochLRWarmup",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 30,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50',
    "loggers": "tensorboard"
}


trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)


best_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/potato_yolonas_run/RUN_20240413_204704_100611/ckpt_best.pth")

average_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/potato_yolonas_run/RUN_20240413_204704_100611/average_model.pth")


#Evaluate(PR Curve)
precision_results = []
recall_results = []

for score_thres in range(0,10):
    test_metrics = DetectionMetrics(
        #Confidence Score
        score_thres=score_thres / 10,
        #IoU 임계값
        iou_thres=0.75,
        top_k_predictions=300,
        num_cls=len(dataset_params['classes']),
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            #NMS 시 Confidence Score, 0.01 이상인 예측만 NMS에 활용
            score_threshold= 0.01,
            nms_top_k=1000,
            max_predictions=300,
            #NMS 시 IoU 임계값, Confidence가 가장 높은 예측과 0.5 이상 겹치면 제거
            nms_threshold=0.5
        )
    )
    test_result = trainer.test(model=best_model,
                               test_loader=test_data,
                               test_metrics_list=test_metrics)

    precision_results.append(test_result.get("Precision@0.75"))
    recall_results.append((test_result.get("Recall@0.75")))

import matplotlib.pyplot as plt
plt.plot(recall_results, precision_results)
plt.show()

#Detect with image
img = cv2.imread("data/potato.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
outputs = best_model.predict(images = img, conf = 0.3)
outputs.show()
