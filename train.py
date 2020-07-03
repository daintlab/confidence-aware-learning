import crl_utils
import utils

import time
import torch
import torch.nn.functional as F

def train(loader, model, criterion_cls, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()

    model.train()
    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        # compute output
        output = model(input)

        # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if args.rank_target == 'softmax':
            conf = F.softmax(output, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif args.rank_target == 'entropy':
            if args.data == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = crl_utils.negative_entropy(output,
                                                    normalize=True,
                                                    max_value=value_for_normalizing)
        # margin
        elif args.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(output), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]

        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        ranking_loss = criterion_ranking(rank_input1,
                                         rank_input2,
                                         rank_target)

        # total loss
        cls_loss = criterion_cls(output, target)
        ranking_loss = args.rank_weight * ranking_loss
        loss = cls_loss + ranking_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.item(), input.size(0))
        ranking_losses.update(ranking_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses,top1=top1))

        # correctness count update
        history.correctness_update(idx, correct, output)
    # max correctness update
    history.max_correctness_update(epoch)

    logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg])
