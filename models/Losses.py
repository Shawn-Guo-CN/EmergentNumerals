import torch

from utils.conf import args

def mask_NLL_loss(prediction, golden_standard, mask, last_eq):
    n_total = mask.sum().item()
    loss = (args.loss_function(prediction, golden_standard) * mask.to(prediction.dtype)).mean()
    eq_cur = prediction.topk(1)[1].squeeze(1).eq(golden_standard).to(prediction.dtype) \
         * mask.to(prediction.dtype)
    n_correct = eq_cur.sum().item()
    eq_cur = eq_cur + (1 - mask.to(prediction.dtype)) * last_eq
    return loss, eq_cur, n_correct, n_total

def mask_NLL_loss_simple(predict, golden, mask, last_eq):
    loss = (args.loss_function(predict, golden) * mask.to(predict.dtype)).mean()
    eq_cur = predict.topk(1)[1].squeeze(1).eq(golden).to(predict.dtype) \
        * mask.to(predict.dtype)
    eq_seq = eq_cur + (1 - mask.to(predict.dtype)) * last_eq
    return loss, eq_cur, eq_seq

def seq_cross_entropy_loss(predict_digits, target, target_mask, target_max_len):
     batch_size = target.shape[1]

     loss = 0
     print_losses = []
     n_correct_tokens = 0
     n_total_tokens = 0
     n_correct_seqs = 0

     seq_correct = torch.ones((1, batch_size), device=args.device)
     eq_vec = torch.ones((1, batch_size), device=args.device)

     for t in range(target_max_len):
         mask_loss, eq_vec, n_correct, n_total = mask_NLL_loss(
             predict_digits[t],
             target[t],
             target_mask[t],
             eq_vec
         )
         loss += mask_loss
         print_losses.append(mask_loss.mean().item())
         n_total_tokens += n_total
         n_correct_tokens += n_correct
         seq_correct = seq_correct * eq_vec

     seq_correct = seq_correct.squeeze(0)
     n_correct_seqs = seq_correct.sum().item()

     tok_acc = round(float(n_correct_tokens) / float(n_total_tokens), 6)
     seq_acc = round(float(n_correct_seqs) / float(batch_size), 6)

     return loss, print_losses, seq_correct, tok_acc, seq_acc
