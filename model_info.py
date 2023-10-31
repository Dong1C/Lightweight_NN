import torch
import matplotlib.pyplot as plt 
import models

# set the models
model_classes = {
    # "squeezenet": models.squeezenet_model,
    "mobilenet": models.mobilenet_model,
    # "xceptionnet": models.xceptionnet_model,
    # "shufflenet": models.shufflenet_model
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model_names = []
    model_parameters = []

    for model_name, model in model_classes.items():
        model_parameters_count = count_parameters(model)

        model_names.append(model_name)
        model_parameters.append(model_parameters_count)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, model_parameters)
    plt.xlabel("Model")
    plt.ylabel("Number of Parameters")
    plt.title("Model Parameter Count")
    plt.xticks(rotation=45)

    # 显示柱状图
    plt.show()
