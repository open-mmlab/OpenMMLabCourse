# CIFAR-10 Evaluation Example

## Single process evaluation

```bash
python cifar10_eval.py
```

## Multiple processes evaluation with torch.distributed

```bash
python cifar10_eval_torch_dist.py
```

## Multiple processes evaluation with MPI4Py

```bash
mpirun -np 3 python3 cifar10_eval_mpi4py.py
```
