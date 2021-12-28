import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name = 'embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="emb."):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# loss = loss_fct(cos_sim, labels)+ config.alpha*compute_kl_loss(z1,z2)
# tmp_loss += loss.item()
# config.scaler.scale(loss).backward()
# config.scaler.step(self.optimizer)
# config.scaler.update()
# fgm.attack()
# loss_adv = self.model(ids, mask=mask, token_type_ids=token_type_ids, batch_size=batch_size,
#                      num_sent=num_sent)
# config.scaler.scale(loss_adv).backward()
# config.scaler.step(self.optimizer)
# config.scaler.update()
# fgm.restore()
# self.optimizer.zero_grad()
# self.scheduler.step()

