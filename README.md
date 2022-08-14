# [Kaggle SIIM COVID19 DETECTION](https://www.kaggle.com/competitions/siim-covid19-detection)
![competions-banner](/misc/banner.png)
This is my repository for attending kaggle SIIM-FISABIO-RSNA COVID-19 Detection

## Evaluation
> The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5. Note that the linked document describes VOC 2012, which differs in some minor ways (e.g. there is no concept of "difficult" classes in VOC 2010). The P/R curve and AP calculations remain the same. In this competition, we are making predictions at both a study (multi-image) and image level.

## Timeline
> - May 17, 2021 - Start Date.
> - August 2, 2021 - Entry Deadline. You must accept the competition rules before this date in order to compete.
> - August 2, 2021 - Team Merger Deadline. This is the last day participants may join or merge teams.
> - August 9, 2021 - Final Submission Deadline (11:59 PM UTC).
## My Result (Public & Private Score)

| Date  | Public | Private |
| ----- | ------ | ------- |
| 30/06 | 0.371  | 0.341   |
| 03/07 | 0.489  | 0.484   |
| 03/07 | 0.492  | 0.484   |
| 07/07 | 0.502  | 0.504   |
| 08/07 | 0.518  | 0.506   |
| 11/07 | 0.529  | 0.524   |
| 11/07 | 0.534  | 0.528   |
| 13/07 | 0.537  | 0.523   |
| 13/07 | 0.540  | 0.532   |
| 15/07 | 0.549  | 0.547   |
| 30/07 | 0.558  | 0.547   |
| 01/08 | 0.567  | 0.557   |
| 03/08 | 0.583  | 0.574   |

Place 829th out of 1,324 teams with 91 sub


## What I could have done better during Competitions.

### Tracking
- Starting from baseline without any techniques applying could help me track how score can improve
- Naming and recording experiment are vital.
    - This could be fixed using **wandb.config** to keep tracking of experiments.
    - **Hydra** could have fixed this issue easily.

### Maintainable and flexing code 
- Write a cleaner code which can help with ease of maintaining code.
- **GlobalConfing** class should have been implemented instead of having config all over the places.
- DataFrame at the start of competition is quite **messy**. (Could have been better and more readable than this)
- code on colab should be identical because it had given unstable results.
- problem solving takes too much time + code runs slow in most cases

### Taking note
- Taking note from what you read in Discussion.
- Keep all the paper that you found useful into one place

## Mistaken
- Using GroupKFold at the first without run random seed again gave me a painful dataleaking.
- I have not checked the how StratifiedGroupKFold acctually works but luckily it did not cause me any problem.
- Start with the biggest model as possible is not quite a great start.
    - This made me wasting a lot a time by meaningless training time.
- Looking back to the similar past competition helped but I tended to looking for key winning 
  and **not reading the whole solution**.
- Sticking to one tool is okay but be able to implementing all tool that could boost the score is better.
   - In this case, MMDet could have gave me a better result but I stick to detectron2 which resulted worse
     than yolov5. This probably occured due to a unstable and inflexibility of framework.
     But what matter the most was myself being coward thinking that mmdet is taking too much time and not worth.
     Which is actually the opposite.
   - I took so much time on finding the best optimizer because I believed it will help with regulization and time saving.
     Turned out it took to much
- At the last moment, I found that I had mistakenly calculated batchsize when using gradient accumulation when logging.
  In training, it did not occur any bug or anything but it had been showing the wrong loss number at the whole time.
- TTA was possibly incorrectly implemented since it did not boost score and get worse some time. 
- I noticed that weark class were indeterminate, atypical but I did not know how to deal with them.
- I spent too much time on augmentations

## What I learned from winner solution.
### Study level
- Auxility loss could be more than 1.
- Adding more different augmentations.
- Resize image possibly needed to be applied **first**.
- DiceLoss worked as aux loss well as well.
- Croping lung in training state.
- implement segmentation network **(DeepLabV3, Unet, Linknet)**.
- Swin transformer is quite powerful.
- ResNet is quite unexpected but it is still working just great.
- implementing aux loss inside Swin Transformer is also possible.
- Pretrained with similar chest image gave a big boost (Chexpert).
- Using script might be a better choice for maintainance.
- Training with 5 classes (negative, typical, indeterminate, atypical, **none**).
- BCE seems to be better that CE some cases.
- CE is still okay to use
- Predicted probability does not need to combine to 1.
- Train more different kind of models.
  - the more variant of model the better
  - different augmentations on different models
  - this gave a model robustness
- Optimizer did not have to be fancy Adam w.orks just fine.
- CosineAnnealingLR is popular choice.
- Increasing image size over 512 does not improve the score.
- Example of Augmentations that worked image level
   - RandomResizedCrop, ShiftScaleRotate, HorizontalFlip, VerticalFlip,
     Blur, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, Cutout
     (from 1st solution)
- Sleep and exercise more
- Add lung segmentation to channel


## Image level
- predicted none class from 5 classes added to image level prediction is allowed and possibly give a boost in my case
    - I tried applying negative class instead of none class for late sub it give me 0.01 in LB and 0.004 in private (from 0.574 and 0.583 repectively)
- removing too big and too small box somehow give a boost but need to be implemented correctly
- 2 state model seemed to work very well such as FasterRCNN FPN
- EfficientDet also worked well
- example of Augmentations that worked image level
  - Scale, RandomResizedCrop, Rotate(maximum 10 degrees), HorizontalFlip, VerticalFlip,
     Blur, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, Cutout, Mosaic, Mixup
     (from 1st solution)
- Mixup is working well
- Unlike Study level, increasing image size on image level gave a better result
- **detecction_conf = detection_conf * (1-image none prediction)^0.4**  
  **none_pred = cl_none_pred0.7 + (1 - image_conf_max)0.3**  
   (I have seen this formula in past past competition but
   I did not quite understand this is much clearer took from 3rd solution)

## Task
- [ ] write run_inference.py