import torch
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader

from mmeval import Accuracy


def get_eval_dataloader():
    dataset = tv.datasets.CIFAR10(
        root='./',
        train=False,
        download=True,
        transform=tv.transforms.ToTensor())
    return DataLoader(dataset, batch_size=1)


def get_model(pretrained_model_fpath=None):
    model = tv.models.resnet18(num_classes=10)
    if pretrained_model_fpath is not None:
        model.load_state_dict(torch.load(pretrained_model_fpath))
    return model.eval()


eval_dataloader = get_eval_dataloader()
model = get_model()
accuracy = Accuracy(topk=(1, 3))

with torch.no_grad():
    for images, labels in tqdm.tqdm(eval_dataloader):
        predicted_score = model(images)
        accuracy.add(predictions=predicted_score, labels=labels)

print(accuracy.compute())
accuracy.reset()
