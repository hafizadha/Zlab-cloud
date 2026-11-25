"""Stop-conditioned arm-head fitting; normal policy inference remains unchanged."""
import torch
from tensordict import TensorDict
from scripts.restricted_actor import penultimate_features

def update_stop_arms(actor, anchor, term, raw):
    from src.tasks.velocity.config.z01_std_add_head.ball_sprint import STANDING_ARM_POSE
    names=list(term.target_names);ids=[names.index(n) for n in STANDING_ARM_POSE]
    scale=torch.as_tensor(term.scale,device=raw.device).expand(len(raw),-1) if torch.as_tensor(term.scale).ndim<2 else term.scale[:1].expand(len(raw),-1)
    offset=torch.as_tensor(term.offset,device=raw.device).expand_as(scale) if torch.as_tensor(term.offset).ndim<2 else term.offset[:1].expand_as(scale)
    scale=scale[:,ids];offset=offset[:,ids]
    if (scale<=0).any() or not torch.isfinite(scale).all():raise ValueError('Invalid arm PD scales.')
    # Verified actor schema: velocity0:3, gravity6:9, command96:99.
    stopped=(raw[:,96:98].norm(dim=1)<=.25)&(raw[:,:2].norm(dim=1)<.3)&(raw[:,8]<-.9)
    running=raw[:,96]>=1.5
    if not stopped.any():return {'stop_arm_aux_states':0.}
    td=TensorDict({'actor':raw},batch_size=[len(raw)])
    with torch.no_grad():
        features=penultimate_features(actor,td).detach()
        before=actor(td,stochastic_output=False)[:,ids]*scale+offset
        running_target=anchor(td,stochastic_output=False)[:,ids]*scale+offset
    desired=torch.tensor(list(STANDING_ARM_POSE.values()),device=raw.device)
    head=actor.mlp[-1];weight=head.weight;bias=head.bias
    prediction=(features @ weight[ids].T+bias[ids])*scale+offset
    loss=(prediction[stopped]-desired).square().mean()
    if running.any():loss=loss+2*(prediction[running]-running_target[running]).square().mean()
    gw,gb=torch.autograd.grad(loss,(weight,bias));norm=torch.sqrt(gw.square().sum()+gb.square().sum()).clamp_min(1.)
    oldw=weight.detach().clone();oldb=bias.detach().clone();retained=1.
    with torch.no_grad():
        for _ in range(18):
            weight.copy_(oldw);bias.copy_(oldb)
            weight[ids]-=.05*retained*gw[ids]/norm;bias[ids]-=.05*retained*gb[ids]/norm
            after=(features @ weight[ids].T+bias[ids])*scale+offset
            if torch.isfinite(after).all() and (after-before).abs().max()<=.03:break
            retained*=.5
        else:weight.copy_(oldw);bias.copy_(oldb);retained=0.
    return {'stop_arm_aux_loss':float(loss.detach()),'stop_arm_aux_states':float(stopped.sum()),'stop_arm_aux_fraction':retained}
