import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm.auto import tqdm
import plots

## Train

#### Constrastive loss

class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, pce, temperature, contrast_mode='one'):
        super().__init__()
        self.pce = pce
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        """Compute loss for model. If `labels` is None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz]
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        # Contrast count is also the number of views
        contrast_count = features.shape[1]
        # Contrast feature [bsz * n_views, ...]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Contrast targets (instance and class based)
        # tile masks [bsz * anchor_count, bsz * contrast_count]
        eye = torch.eye(batch_size, dtype=torch.float32, device=device)

        # Instance mask
        inst_mask = eye.repeat(anchor_count, contrast_count)

        # Class mask
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        class_mask = torch.eq(labels, labels.T).float().to(device)
        class_mask = class_mask.repeat(anchor_count, contrast_count)

        # Positive and negative mask
        pos_mask = torch.maximum(inst_mask, class_mask)
        neg_mask = 1 - pos_mask

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(inst_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        inst_mask *= logits_mask
        class_mask *= logits_mask
        pos_mask *= logits_mask
        neg_mask *= logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob (a.k.a cross entropy) for each positive pair against all negative pairs and itself
        exp_logits = torch.exp(logits)
        sum_exp_logits = torch.sum(exp_logits * logits_mask, dim=1, keepdim=True)
        neg_sum_exp_logits = torch.sum(exp_logits * neg_mask, dim=1, keepdim=True)

        # Cross entropies
        log_probs = logits - torch.log(sum_exp_logits)
        partial_log_prob = logits - torch.log(neg_sum_exp_logits + exp_logits)

        if self.pce:
            # Instance cross entropy
            inst_log_prob = (inst_mask * partial_log_prob).sum(1) / inst_mask.sum(1)
            # Class partial cross entropy
            class_log_prob = (class_mask * partial_log_prob).sum(1) / (class_mask.sum(1) + 1)
        else:
            # Instance cross entropy
            inst_log_prob = (inst_mask * log_probs).sum(1) / inst_mask.sum(1)
            # Class cross entropy
            class_log_prob = (class_mask * log_probs).sum(1) / (class_mask.sum(1) + 1)

        # loss
        loss = -(inst_log_prob + class_log_prob).mean()

        return loss


#### Loop

def train_loop(args, data_loader, model, opt=None):
    all_losses, all_label, all_pred = [], [], []
    pbar = tqdm(data_loader, 'train' if opt is not None else 'test', leave=False)
    training = opt is not None
    con_loss_fn = ConLoss(args.pce, args.temp)
    model.cuda()
    for img_views, labels in pbar:
        # Set to GPU
        img_views = [imgs.cuda() for imgs in img_views]
        labels = labels.cuda()

        # Train/test setup
        if training:
            model.train()
            opt.zero_grad()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        # Features
        feat_views = model.feats(img_views)
        feats = feat_views[0]

        # Contrastive representation?
        con_loss = 0
        if args.contrast and training:
            assert not args.fix_feats and len(feat_views) > 1
            # Projected features
            project_views = model.project(feat_views)
            project_views_t = torch.cat([p.unsqueeze(1) for p in project_views], dim=1)
            con_loss = con_loss_fn(project_views_t, labels)

            # Detach features so optimizing classifier doesn't affect it
            feats = feats.detach()

        # Cross entropy
        out = model.fc(feats)
        ce_loss = F.cross_entropy(out, labels)

        # Total loss
        loss = con_loss + ce_loss

        # Prediction
        pred = torch.argmax(out, dim=1)

        # Backprop?
        if training:
            loss.backward()
            opt.step()

        # Record
        all_losses.append(loss.item())
        all_label.extend(labels.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())

    return all_losses, all_label, all_pred


#### Flow

def train(args, model, train_loader, test_loader):
    # Optimizer
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=1e-4)
    # Scheduler
    if args.steplr is not None:
        assert not args.cosine
        scheduler = optim.lr_scheduler.StepLR(opt, args.steplr, gamma=0.1)
    elif args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    else:
        scheduler = None

    # Setup
    metrics = plots.Metrics()
    pbar = tqdm(range(args.epochs), 'epochs')
    for _ in pbar:
        # Train
        try:
            train_metrics = train_loop(args, train_loader, model, opt)
            test_metrics = train_loop(args, test_loader, model)
        except KeyboardInterrupt:
            break

        # Record metrics
        metrics.epoch_append_data('train', *train_metrics)
        metrics.epoch_append_data('test', *test_metrics)

        # Progress bar update
        pbar.set_postfix_str(metrics.epoch_str())

        # Save model weights
        torch.save(model.state_dict(), f'{args.outdir}/model.pt')
        torch.save(model.encoder.state_dict(), f'{args.outdir}/encoder.pt')

        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    return metrics
