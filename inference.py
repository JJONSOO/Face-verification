import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from deepface.commons import functions, distance as dst
from backbones import get_model
import pandas as pd
import tqdm
from PIL import Image

@torch.no_grad()
def inference(weight, name):

    submission = pd.read_csv("./sample_submission.csv")
    def cos_sim(a, b):
        return F.cosine_similarity(a, b)
    left_test_paths = list()
    right_test_paths = list()
    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])
    results = []
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight,map_location = 'cuda:0'))
    net.eval()
    net = net.cuda(0)


    for left_test_path, right_test_path in tqdm.tqdm(zip(left_test_paths, right_test_paths)):
        # img1 = cv2.imread('./test/' + left_test_path + '.jpg')
        # img1 = cv2.resize(img1, (112, 112))
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = np.transpose(img1, (2, 0, 1))
        # img1 = torch.from_numpy(img1).unsqueeze(0).float()
        # img1.div_(255).sub_(0.5).div_(0.5)
        # img1 = img1.cuda(0)


        img_left = Image.open("./test/" + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img_left = np.array(img_left)
        img_left = functions.preprocess_face(img_left, target_size = (112, 112), detector_backend = 'opencv', enforce_detection = False)
        img_left = torch.from_numpy(img_left).float()
        img_left = img_left.cuda(0)
        img_left = img_left.permute([0,3,1,2])

        # img2 = cv2.imread('./test/' + right_test_path + '.jpg')
        # img2 = cv2.resize(img2, (112, 112))
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # img2 = np.transpose(img2, (2, 0, 1))
        # img2 = torch.from_numpy(img2).unsqueeze(0).float()
        # img2.div_(255).sub_(0.5).div_(0.5)
        # img2 = img2.cuda(0)

        img_right = Image.open("./test/" + right_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img_right = np.array(img_right)
        img_right = functions.preprocess_face(img_right, target_size = (112, 112), detector_backend = 'opencv', enforce_detection = False)
        img_right = torch.from_numpy(img_right).float()
        img_right = img_right.cuda(0)
        img_right = img_right.permute([0,3,1,2])


        
        feat1 = net(img_left)
        feat2 = net(img_right)
        cosin_sim = cos_sim(feat1, feat2)
        results.append(cosin_sim)
        temp = []
        for i in results:
            temp.append(float(i))
        
    submission = pd.read_csv("./sample_submission.csv") 
    submission['answer'] = temp
    submission.to_csv('./r50_submission.csv', index=False)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    
    args = parser.parse_args()
    inference(args.weight, args.network)