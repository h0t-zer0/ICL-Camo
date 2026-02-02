"""
Modified according to
https://github.com/GewelsJI/SINet-V2/blob/main/utils/fps.py
https://github.com/yuhuan-wu/MobileSal/blob/master/speed_test.py
https://github.com/CRRCOO/FINet/blob/main/speed_test.py
""" 

import torch, time
import numpy as np


# Define model
from Model.ICLCamo import ICLCamo
model = ICLCamo()

# Define numpy input tensor
bs = 1
test_img = np.random.random((bs, 3, 392, 392)).astype('float32')
test_ref = np.random.random((bs, 3, 392, 392)).astype('float32')
test_mask = np.random.random((bs, 1, 392, 392)).astype('float32')

# Define pytorch & jittor input tensor
pytorch_test_img = torch.Tensor(test_img).cuda()
pytorch_test_ref = torch.Tensor(test_ref).cuda()
pytorch_test_mask = torch.Tensor(test_mask).cuda()

# Run [turns] times to get average time
turns = 100

# Define pytorch & jittor model
pytorch_model = model.cuda()
pytorch_model.eval()

# Pytorch warm up and one time forward time test
for i in range(10):
	pytorch_result = pytorch_model(pytorch_test_img, pytorch_test_ref, pytorch_test_mask)
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
	pytorch_result = pytorch_model(pytorch_test_img, pytorch_test_ref, pytorch_test_mask)
# Only when ''torch.cuda.synchronize()'' is called, the time measurement is accurate
torch.cuda.synchronize()
end = time.time()
tc_time = round((end - sta) / turns, 5)  # Run [turns] times to get average time
tc_fps = round(bs * turns / (end - sta), 0)  # Calculate FPS
print(f"- Pytorch forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")
