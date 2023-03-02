import torch
import torchvision as tv
import tqdm
from mpi4py import MPI
from torch.utils.data import DataLoader, DistributedSampler

from mmeval import Accuracy


def get_eval_dataloader(rank=0, num_replicas=1):
    dataset = tv.datasets.CIFAR10(
        root='./',
        train=False,
        download=True,
        transform=tv.transforms.ToTensor())
    dist_sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank)
    data_loader = DataLoader(dataset, batch_size=1, sampler=dist_sampler)
    return data_loader, len(dataset)


def get_model(pretrained_model_fpath=None):
    model = tv.models.resnet18(num_classes=10)
    if pretrained_model_fpath is not None:
        model.load_state_dict(torch.load(pretrained_model_fpath))
    return model.eval()


def eval_fn(rank, process_num):
    torch.cuda.set_device(rank)
    eval_dataloader, total_num_samples = get_eval_dataloader(rank, process_num)
    model = get_model('./cifar10_resnet18.pth').cuda()
    accuracy = Accuracy(topk=(1, 3), dist_backend='mpi4py')

    with torch.no_grad():
        for images, labels in tqdm.tqdm(eval_dataloader, disable=(rank != 0)):
            predicted_score = model(images.cuda()).cpu()
            accuracy.add(predictions=predicted_score, labels=labels)

    print(accuracy.compute(size=total_num_samples))
    accuracy.reset()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    eval_fn(comm.Get_rank(), comm.Get_size())
