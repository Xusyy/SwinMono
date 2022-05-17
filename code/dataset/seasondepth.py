import os
import cv2

from dataset.base_dataset import BaseDataset


class seasondepth(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/', 
                 is_train=True, dataset='seasondepth', scale_size=None):
        super().__init__()        

        self.scale_size = scale_size
        
        self.is_train = is_train
        self.data_path = data_path    

        self.image_path_list = []
        self.depth_path_list = []
        txt_path = os.path.join(filenames_path, 'seasondepth')
        
        if is_train:
            txt_path += '/train_list.txt'
        else:
            txt_path += '/test_list.txt'        
        
        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'        
        print("Dataset :", dataset)
        print("# of %s images: %d" % (phase, len(self.filenames_list)))


    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-1]

        image = cv2.imread(img_path)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
               
        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[1], self.scale_size[0]))
            depth = cv2.resize(depth, (self.scale_size[1], self.scale_size[0]))
          
        if self.is_train:
            image, depth = self.augment_training_data(image, depth)        
        else:
            image, depth = self.augment_test_data(image, depth)
        

        return {'image': image, 'depth': depth, 'filename': filename}
