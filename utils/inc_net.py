import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinear_RanPAC
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from einops import einsum
import os
import json
import torchvision.transforms as transforms
from utils.toolkit import get_attribute
import difflib
from PIL import Image
import random
from peft import LoraConfig, AdaptionPromptConfig, get_peft_model, PromptTuningConfig, PromptTuningInit
random.seed(1993)


def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()

    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name == 'clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name == 'openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))

    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(
                        self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(
                    nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(
            args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(
                    nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)

    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out


class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.convnet, self.preprocess, self.tokenizer = get_convnet(
            args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(
                    nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)

    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):

        image_features, text_features, logit_scale = self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(
            self.args, True)


class Area(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)

        self.model, self.preprocess, self.tokenizer = get_convnet(
            args, pretrained)
        self.class_name = 'Area'
        self.args = args
        self.K = get_attribute(args, "K", 16)
        self.class_names = None
        self.visual_adapter = nn.Linear(512, 512, bias=False)
        self.freeze(self.model)
        self.textual_adapter = nn.Linear(512, 512, bias=False)
        self.textual_S = nn.ModuleList()
        self.visual_S = nn.ModuleList()
        # class stat
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        self.class_mean_list = []
        self.class_cov_list = []

    def append_S(self, device):
        self.textual_S.append(nn.Linear(512, self.K, bias=False).to(device))
        self.visual_S.append(nn.Linear(512, self.K, bias=False).to(device))
        # If cur_task > 0, new S initialized as previous S
        if len(self.textual_S) > 1:
            self.textual_S[-1].weight.data = self.textual_S[-2].weight.data.clone()
            self.visual_S[-1].weight.data = self.visual_S[-2].weight.data.clone()
        self.visual_S[-1].weight.requires_grad = True
        self.textual_S[-1].weight.requires_grad = True

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(
                    nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.model.encode_image(x)

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def forward(self, image, text_embeddings, visual_basis, textual_basis, cur_task, memory_data=None):
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        if memory_data is not None:
            memory_data = memory_data.to(image.device)
            image_features = torch.cat([image_features, memory_data], dim=0)
        image_features_residual = self.visual_adapter(image_features.detach())
        image_features_evidence = einsum(
            visual_basis, self.visual_S[cur_task](image_features), "C D K, B K -> B C D")
        image_features_residual = image_features_residual.unsqueeze(
            1).expand(-1, textual_basis.shape[0], -1)
        image_features = image_features_residual + image_features_evidence
        textual_features_residual = self.textual_adapter(
            text_embeddings.detach())
        textual_features_evidence = einsum(textual_basis, self.textual_S[cur_task](
            text_embeddings.detach()), "C D K, C K -> C D")
        textual_features = textual_features_residual + textual_features_evidence
        image_features = image_features / \
            (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        textual_features = textual_features / \
            (textual_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits = einsum(image_features, textual_features, "B C D, C D -> B C")
        logit_scale = self.model.logit_scale.exp()
        logits = logits * logit_scale
        probs = logits
        return probs

    def forward_inference(self, image, text_embeddings, visual_basis, textual_basis, cur_task, memory_data=None):
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        if memory_data is not None:
            memory_data = memory_data.to(image.device)
            image_features = torch.cat([image_features, memory_data], dim=0)
        image_features_residual = self.visual_adapter(image_features.detach())
        image_features_evidence = einsum(
            visual_basis, self.visual_S[cur_task](image_features), "C D K, B K -> B C D")
        # B D -> B C D
        image_features_residual = image_features_residual.unsqueeze(
            1).expand(-1, textual_basis.shape[0], -1)
        image_features = image_features_residual + image_features_evidence
        textual_features_residual = self.textual_adapter(
            text_embeddings.detach())
        textual_features_evidence = einsum(textual_basis, self.textual_S[cur_task](
            text_embeddings.detach()), "C D K, C K -> C D")
        textual_features = textual_features_residual + textual_features_evidence
        image_features = image_features / \
            (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        textual_features = textual_features / \
            (textual_features.norm(dim=-1, keepdim=True) + 1e-6)
        logits = einsum(image_features, textual_features, "B C D, C D -> B C")
        logit_scale = self.model.logit_scale.exp()
        logits = logits * logit_scale
        probs = logits
        return probs

    def _get_visual_score(self, image, cur_task):
        image_features = self.model.encode_image(image)
        return self.visual_S[cur_task](image_features)

    def _get_textual_score(self, text, cur_task):
        tokenized_text = self.tokenizer(text).to(
            next(self.model.parameters()).device)
        text_features = self.model.encode_text(tokenized_text)
        return self.textual_S[cur_task](text_features)

    def re_initiate(self):
        print('re-initiate model')
        self.model, self.preprocess, self.tokenizer = get_convnet(
            self.args, True)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def analyze_mean_cov(self, features, labels):
        print(labels)
        label = torch.sort(torch.unique(labels))[0]
        print("analyzing mean and cov")
        print("number of classes:", label.shape[0])
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * \
                torch.eye(class_data.shape[-1], device=class_data.device)
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def update_stat(self, known_classes, total_classes, train_loader, device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            # vecs_512 = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features = self.visual_forward_(inputs)
                image_features = image_features / \
                    image_features.norm(dim=-1, keepdim=True)

                vecs.append(image_features)
                labels.append(targets)

            vecs = torch.cat(vecs)
            labels = torch.cat(labels)

            mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True)
                           for i in range(known_classes, total_classes)], dim=0)
            center_vecs = torch.cat([vecs[labels == i] - mu[i - known_classes]
                                    for i in range(known_classes, total_classes)], dim=0)
            cov_inv = center_vecs.T @ center_vecs / (center_vecs.shape[0] - 1)
            tmp = (center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov(
            ).trace() * torch.eye(center_vecs.shape[1]).to(device)
            tmp = tmp.to("cpu")
            tmp = torch.linalg.pinv(tmp)
            tmp = tmp.to(device)
            cov_inv = center_vecs.shape[1] * tmp
            if not hasattr(self, 'mu'):
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (known_classes/total_classes)*self.cov_inv + (total_classes-known_classes)/total_classes*cov_inv + ((known_classes/total_classes)*(total_classes-known_classes) /
                                                                                                                                   total_classes**2)*(self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
                self.mu = torch.cat([self.mu, mu])
            ps = torch.ones(self.mu.shape[0]).to(
                device) * 1. / self.mu.shape[0]
            self.W = torch.einsum('nd, dc -> cn', self.mu, self.cov_inv)
            self.b = ps.log() - torch.einsum('nd, dc, nc -> n',
                                             self.mu, self.cov_inv, self.mu) / 2

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self._expand_token(
            self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        if self.visual.attn_pool is not None:
            if self.visual.attn_pool_contrastive is not None:
                x = self.visual.ln_post(x)
                tokens = self.visual.attn_pool(x)
                if self.visual.attn_pool_type == 'parallel':
                    pooled = self.visual.attn_pool_contrastive(x)
                else:
                    assert self.visual.attn_pool_type == 'cascade'
                    pooled = self.visual.attn_pool_contrastive(tokens)
            else:
                x = self.visual.attn_pool(x)
                x = self.visual.ln_post(x)
                pooled, tokens = self.visual._global_pool(x)
        elif self.visual.final_ln_after_pool:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        else:
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        return pooled

    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)
