import argparse
import models

# 定义模型名称与对应的PyTorch模型类
model_classes = {
    "squeezenet": models.squeezenet_model,
    "mobilenet": models.mobilenet_model,
    "xceptionnet": models.xceptionnet_model,
    "shufflenet": models.shufflenet_model
}



if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a selected model on MTB')
    parser.add_argument('--model', choices=model_classes.keys(), required=True, help='Select a model (squeezenet, mobilenet, xceptionnet, shufflenet)')
    parser.add_argument('--epoches', default=10, help='specify the epoches of the training')
    args = parser.parse_args()
    print(args.model)
    print(args.epoches)
