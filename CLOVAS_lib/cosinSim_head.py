import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import FocalLoss, BinaryDiceLoss
from scipy.ndimage import gaussian_filter

    
class ConsineSimHead(nn.Module):
    def __init__(
            self,
            img_size
    ):
        super(ConsineSimHead, self).__init__()

        self.image_size = img_size
        self.binary_focal_loss = FocalLoss()
        self.binary_dice_loss = BinaryDiceLoss()

    def forward_test(self, inputs,topk=None,sigma = 4,specify_ids=None):
        out  = self.forward(inputs)
        out["text_probs"] = (out["text_probs"]/0.07).softmax(-1)[:,0,0]
        oa_similarity_map_list = out["similarity_map_list"]
        anomaly_map_list = []
        for similarity_map in oa_similarity_map_list:
            similarity_map = similarity_map.permute(0,2,3,1)
            anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
            anomaly_map_list.append(anomaly_map)
        anomaly_map = torch.stack(anomaly_map_list)
    
        anomaly_map = anomaly_map.sum(dim = 0)
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )

        out.update({"anomaly_map":anomaly_map})
        return out

    def forward(self, feats):
        patch_features = feats['patch_features']
        global_embedding = feats['global_embedding'] # b,dim

        oa_text_embedding = feats['oa_text_embedding'] # 2,512
        out = {}
        # for binary anomaly segmentation
        # text_probs = global_embedding.unsqueeze(1) @ oa_text_embedding.permute(0, 2, 1) # b,1,2
        # text_probs = text_probs[:, 0, ...]/0.07
        logit_scale = self.logit_scale.exp()
        if len(oa_text_embedding.shape)==2:
            # text_probs = logit_scale * global_embedding @ oa_text_embedding.t() # b,2
            oa_text_embedding=oa_text_embedding.unsqueeze(0)
            text_probs = global_embedding.unsqueeze(1) @ oa_text_embedding.permute(0, 2, 1)
        else:
            logits = []
            for t_feat,i_feat in zip(oa_text_embedding,global_embedding):
                logits.append(logit_scale * i_feat @ t_feat.t())
            text_probs = torch.stack(logits,dim=0)
        # text_probs = text_probs[:, ...]/0.07
        similarity_map_list = []
        # similarity_map_list.append(similarity_map)
        for idx, patch_feature in enumerate(patch_features):
            patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
            similarity, _ = self.compute_similarity(patch_feature, oa_text_embedding[0])
            similarity_map = self.get_similarity_map(similarity[:, 1:, :], self.image_size).permute(0, 3, 1, 2)
            similarity_map_list.append(similarity_map)
        out.update({"text_probs":text_probs,
                    "similarity_map_list":similarity_map_list
                    })
        return out


    def get_similarity_map(self,sm, shape):
        side = int(sm.shape[1] ** 0.5)
        sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
        sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
        sm = sm.permute(0, 2, 3, 1)
        return sm


    def compute_similarity(self,image_features, text_features, t=2):
        # image_features:[b,1370,768] 
        if len(text_features.shape)==2: # text_features:[2,768]
            prob_1 = image_features[:, :1, :] @ text_features.t()
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
            similarity = feats.sum(-1)

        else: # text_features:[b,2,768]
            prob_1 = image_features[:, :1, :] @ text_features.permute(0,2,1) # [b,1,2]
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[1], image_features.shape[1], image_features.shape[2]
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(b, 1, n_t, c) # [b,1370,2,768]
            similarity = feats.sum(-1) # [b,1370,2]
        return (similarity/0.07).softmax(-1), prob_1

    def forward_train(self,feat,gt):
        out = self.forward(feat)
        similarity_map_list = out['similarity_map_list']

        out['text_probs']=out['text_probs'][:, 0, ...]/0.07
        gt_masks = gt["masks"]
        gt_masks[gt_masks>0] = 1
        labels = gt["labels"]
        losses = {}
        focal_loss = 0
        dice_loss = 0
        for i in range(len(similarity_map_list)):
            focal_loss += self.binary_focal_loss(similarity_map_list[i], gt_masks)
            dice_loss += self.binary_dice_loss(similarity_map_list[i][:, 1, :, :], gt_masks)
            dice_loss += self.binary_dice_loss(similarity_map_list[i][:, 0, :, :], 1-gt_masks)
        image_loss = F.cross_entropy(text_probs, labels.long())
        losses = {
            "loss_binary_focal": focal_loss,
            "loss_binary_dice": dice_loss,
            "loss_constractive":image_loss
        }
        return losses


if __name__ == "__main__":
    pass