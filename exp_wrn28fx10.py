import torch
import torchvision
from nn_utils.models import wide_resnet28x10, wide_resnet28fx10, wide_resnet28nfx10
from nn_utils.attacks import LinfPGDAttack, parameter_presets
from nn_utils.training import train_classifier, \
    AdversarialPerturbationCallback, LRSchedulerCallback, CheckpointCallback, WandBLoggerCallback, CutMixCallback
from nn_utils.misc import OneHotEncoder, CrossEntropyLoss, \
    load_cifar10, split_dataset, create_data_loaders, set_determinism, top1_accuracy
import warnings


def main():
    warnings.filterwarnings('ignore', category=Warning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(params):
        CIFAR10_STATS = {
            'means': (0.4914, 0.4822, 0.4465),
            'stds': (0.2023, 0.1994, 0.2010)
        }
        train_transforms = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]
        test_transforms = []
        target_transform = OneHotEncoder(10, dtype=torch.float16)
        train_ds, test_ds = load_cifar10(train_transforms, test_transforms, target_transform)
        test_ds, val_ds = split_dataset(test_ds, 0.1, seed=42)
        train_loader, test_loader, val_loader = create_data_loaders([train_ds, test_ds, val_ds], 512)

        set_determinism(True, 42)
        model = wide_resnet28fx10(10, **CIFAR10_STATS)
        model.to(device)

        config = {
            'num_epochs': 160,
            'batch_size': 512,
            'loss_fn': 'CrossEntropyLoss',
            'optimizer': 'SGD',
            'base_learning_rate': params['base_lr'],
            'learning_rate_schedulers': ['ConstantLR', 'ConstantLR', 'ConstantLR'],
            'dataset': 'CIFAR10',
            'training_type': 'adv',
            'architecture': 'WideResNet28fx10'
        }
        bias_parameters = [p for n, p in model.named_parameters() if 'bias' in n]
        scale_parameters = [p for n, p in model.named_parameters() if 'scale' in n]
        other_parameters = [p for n, p in model.named_parameters() if not ('bias' in n or 'scale' in n)]
        loss_fn = CrossEntropyLoss(one_hot=True).to(device)
        base_lr = params['base_lr']
        optimizer = torch.optim.SGD([
            {'params': bias_parameters, 'lr': base_lr / 10.},
            {'params': scale_parameters, 'lr': base_lr / 10.},
            {'params': other_parameters}
        ], lr=base_lr, weight_decay=params['weight_decay'], momentum=0.9, nesterov=True)

        def lr_fn(epoch):
            if epoch < 100:  # [1, 100)
                return 1.
            if epoch < 150:  # [100, 150)
                return 1e-1
            return 1e-2  # [150, inf)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
        attack = LinfPGDAttack(model, loss_fn, **parameter_presets['CIFAR10'], random_start=True, device=device)

        def metrics_comparator(metrics, best_metric):
            adv_test_acc = metrics['adv_test_acc']
            if best_metric is None or adv_test_acc > best_metric:
                return adv_test_acc
            return best_metric

        metrics = train_classifier(model, loss_fn, optimizer, train_loader, test_loader, attack,
                                   acc_fn=top1_accuracy(one_hot=True),
                                   callbacks=[
                                       CheckpointCallback(
                                           f'../checkpoints/adv_wresnet28fx10_cutmix_{list(params.values())}',
                                           metric_comparator=metrics_comparator
                                       ),
                                       WandBLoggerCallback('AdvBatchNorm', f'adv-wresnet28fx10-cutmix-{list(params.values())}', config),
                                       CutMixCallback(1.),
                                       AdversarialPerturbationCallback(attack),
                                       LRSchedulerCallback(scheduler)
                                   ],
                                   num_epochs=160, initial_epoch=1)
        return metrics['adv_test_acc']

    best_value = 0.0
    best_config = None
    for lr in [2e-1, 1e-1, 5e-2]:
        for wd in [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            config = {'base_lr': lr, 'weight_decay': wd}
            value = objective(config)
            if value > best_value:
                best_value = value
                best_config = config
    print(best_value, best_config)


if __name__ == '__main__':
    main()
