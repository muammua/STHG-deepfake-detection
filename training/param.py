import torch
import yaml

from detectors import DETECTOR
from thop import profile  # 新增thop库导入

detector_path = './training/config/detector/sthhg1.yaml'

# weights_path = '/root/wxy/DeepfakeDetection/DeepfakeBench-main/logs/training/sthhg1234/test/avg/ckpt_best.pth'
with open(detector_path, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_class = DETECTOR[config['model_name']]
model = model_class(config).to(device)
# try:
#     ckpt = torch.load(weights_path, map_location=device)
#     model.load_state_dict(ckpt, strict=True)
#     print('===> Load checkpoint done!')
# except Exception as e:
#     print(f'Fail to load the pre-trained weights: {e}')


# 计算模型参数量和FLOPs
model.eval()
# example_input = torch.randn(1, 3, config['resolution'], config['resolution']).to(device)
example_input = torch.randn(1, 4, 3, config['resolution'], config['resolution']).to(device)
data_dict = {
    'image': example_input,
    'label': torch.tensor([]).to(device)
}
# 计算FLOPs和参数量
flops, params = profile(model, inputs=(data_dict, True), verbose=False)
print(f"Parameters: {params / 1e6:.2f}M")
print(f"FLOPs: {flops / 1e9:.2f}G")