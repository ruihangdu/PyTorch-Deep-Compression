from compressor import *


def iter_prune(args, data_func=get_loaders, model_class=None, **kwargs):
    trainloader, val_loader, test_loader = data_func(args)

    topk = tuple(sorted(args.topk)) # sort in ascending order

    epoch = 1
    best_loss = sys.maxsize

    if args.arch:
        model = models.__dict__[args.arch](args.pretrained)
        optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr)
        # TODO: check if the optimizer needs momentum and weight decay
        if args.optimizer == 'SGD':
            optimizer.momentum = args.momentum
            optimizer.weight_decay = args.weight_decay
    else:
        assert model_class is not None
        model = model_class(**kwargs)

        optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr)
        # TODO: check if the optimizer needs momentum and weight decay
        if args.optimizer == 'SGD':
            optimizer.momentum = args.momentum
            optimizer.weight_decay = args.weight_decay

        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch'] % args.max_epochs + 1
            best_loss = checkpoint['best_loss']
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                raise e

    print(model.__class__)

    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    compressor = Compressor(model, cuda=args.cuda)

    model.train()
    pct_pruned = 0.0
    scores = [AverageMeter() for _ in topk]

    val_scores = [0.0 for _ in topk]

    while True:
        if epoch == 1:
            new_pct_pruned = compressor.prune()

            logging.info('Pruned %.3f %%' % (100 * new_pct_pruned))

            top_accs = validate(model, val_loader, topk, cuda=args.cuda)

            if new_pct_pruned - pct_pruned <= 0.001 and converged(val_scores, top_accs):
                break

            pct_pruned = new_pct_pruned
            val_scores = top_accs

        for e in range(epoch, args.max_epochs + 1):
            for i, (input, label) in enumerate(trainloader, 0):
                input, label = Variable(input), Variable(label)

                if args.cuda:
                    input, label = input.cuda(), label.cuda()

                optimizer.zero_grad()

                output = model(input)

                precisions = accuracy(output, label, topk)

                for i, s in enumerate(scores):
                    s.update(precisions[i][0], input.size(0))

                loss = criterion(output, label)
                loss.backward()

                compressor.set_grad()

                optimizer.step()

            if e % args.interval == 0:
                checkpoint = {
                    'state_dict': model.module.state_dict()
                    if args.cuda else model.state_dict(),
                    'epoch': e,
                    'best_loss': max(best_loss, loss.item()),
                    'optimizer': optimizer.state_dict()
                }

                save_checkpoint(checkpoint, is_best=(loss.item()<best_loss))

            if e % 30 == 0:
                # TODO: currently manually adjusting learning rate, could be changed to user input
                lr = optimizer.lr * 0.1
                adjust_learning_rate(optimizer, lr, verbose=True)

        epoch = 1

    test_topk = validate(model, test_loader, topk, cuda=args.cuda)


def main():
    from lenet_300 import LeNet_300_100 as Model
    args = parse_args()

    iter_prune(args, data_func=get_mnist_loaders, model_class=Model)


if __name__ == '__main__':
    main()