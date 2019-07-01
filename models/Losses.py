from utils.conf import args

def mask_NLL_loss(prediction, golden_standard, mask, last_eq):
    n_total = mask.sum().item()
    loss = (args.loss_function(prediction, golden_standard) * mask.to(prediction.dtype)).mean()
    eq_cur = prediction.topk(1)[1].squeeze(1).eq(golden_standard).to(prediction.dtype) \
         * mask.to(prediction.dtype)
    n_correct = eq_cur.sum().item()
    eq_cur = eq_cur + (1 - mask.to(prediction.dtype)) * last_eq
    return loss, eq_cur, n_correct, n_total