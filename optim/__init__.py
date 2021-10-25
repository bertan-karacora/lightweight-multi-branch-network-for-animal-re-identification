import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from .n_adam import NAdam
from .warmup_scheduler import WarmupMultiStepLR
from .warmup_cosine_scheduler import WarmupCosineAnnealingLR


def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    if args.model in ['PCB', 'PCB_v', 'PCB_conv']:
        ignored_params = []
        for i in range(args.parts):
            name = 'classifier' + str(i)
            c = getattr(model, name)
            ignored_params = ignored_params + list(map(id, c.parameters()))

        ignored_params = tuple(ignored_params)

        base_params = filter(lambda p: id(
            p) not in ignored_params, model.model.parameters())

        if args.pcb_different_lr == 'True':
            print('PCB different lr')
            if args.optimizer == 'SGD':
                optimizer_pcb = optim.SGD([
                    {'params': base_params, 'lr': 0.1 * args.lr},
                    {'params': model.model.classifier0.parameters(), 'lr': args.lr},
                    {'params': model.model.classifier1.parameters(), 'lr': args.lr},
                    {'params': model.model.classifier2.parameters(), 'lr': args.lr},
                    {'params': model.model.classifier3.parameters(), 'lr': args.lr},
                    {'params': model.model.classifier4.parameters(), 'lr': args.lr},
                    {'params': model.model.classifier5.parameters(), 'lr': args.lr},

                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
                return optimizer_pcb
            elif args.optimizer == 'ADAM':
                params = []
                for i in range(args.parts):
                    name = 'classifier' + str(i)
                    c = getattr(model.model, name)
                    params.append({'params': c.parameters(), 'lr': args.lr})
                params = [{'params': base_params,
                           'lr': 0.1 * args.lr}] + params

                optimizer_pcb = optim.Adam(params, weight_decay=5e-4)

                return optimizer_pcb
            else:
                raise('Optimizer not found, please choose adam or sgd.')

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
        }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = NAdam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer, last_epoch):
    milestones = args.decay_type.split('_')
    milestones.pop(0)
    milestones = list(map(lambda x: int(x), milestones))
    if args.cosine_annealing:
        scheduler = lrs.CosineAnnealingLR(optimizer, float(args.epochs), last_epoch=last_epoch)
        return scheduler

    elif args.w_cosine_annealing:
        scheduler = WarmupCosineAnnealingLR(optimizer, multiplier=1, warmup_epoch=int(round(args.epochs / 5.0)), min_lr=args.lr / 1000, epochs=args.epochs, last_epoch=last_epoch)
        return scheduler

    scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=args.gamma, warmup_factor=0.01, warmup_iters=int(round(args.epochs / 5.0)), warmup_method=args.warmup, last_epoch=last_epoch)
    return scheduler
