import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class iMaterialistDataset(Dataset):
    """
    Args:
        dataset_df (pd.DataFrame): dataframe with `img_path` and `rle mask` columns
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            dataset_df,
            class_info,
            img_dir,
            img_size=512,
            resize=False,
            classes=None,
            augmentation=None,
            preprocessing=None
    ):
        self.df = dataset_df
        self.img_dir = img_dir
        self.img_size = img_size
        self.resize = resize

        # all classes from meta
        self.CLASSES = list(class_info['cat_name'])
        self.CLASS_MAP = dict(class_info[['cat_name', 'id']].values)
        self.n_classes = len(self.CLASSES)

        if classes is None:
            classes = self.CLASSES

        # convert str names to class values on masks
        self.class_values = [self.CLASS_MAP[cls.lower()] for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        # load image
        img_path = self.df.index[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, img_path)))
        if self.resize:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            if len(image.shape) < 3:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        # build mask
        mask = self.load_mask(img_path)

        # extract certain classes from mask (e.g. cars)
        mask = mask[..., np.array(self.class_values)].astype('float')
     
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.df)
    
    def load_mask(self, image_id):
        info = self.df.loc[image_id]

        mask = np.zeros((self.img_size, self.img_size, self.n_classes), dtype=np.uint8)

        for m, (annotation, label) in enumerate(zip(info['EncodedPixels'], info['CategoryId'])):
            sub_mask = np.full(info['Height'] * info['Width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['Height'], info['Width']), order='F')
            sub_mask = cv2.resize(sub_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask[:, :, label] += sub_mask
                
        mask = np.clip(mask, 0, 1).astype('uint8')

        return mask
