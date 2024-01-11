# The code below is a condensed single-file version of the ATISS architecture
# definition from https://github.com/nv-tlabs/ATISS . Original terms apply.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
# Andreas Geiger, Sanja Fidler


import numpy as np
import torch
import torch.nn as nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask
from torchvision import models
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def cross_entropy_loss(pred, target):
    """Cross entropy loss."""
    B, L, C = target.shape
    loss = torch.nn.functional.cross_entropy(
        pred.reshape(-1, C), target.reshape(-1, C).argmax(-1), reduction="none"
    ).reshape(B, L)

    return loss


def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation that prevents
    overflow.
    """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def dmll(pred, target, log_scale_min=-7.0, num_classes=256):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].

    Code adapted
    from https://github.com/idiap/linear-transformer-experiments/blob/0a540938ec95e1ec5b159ceabe0463d748ba626c/image-generation/utils.py#L31

    Arguments
    ----------
        pred (Tensor): Predicted output (B x L x T)
        target (Tensor): Target (B x L x 1).
        log_scale_min (float): Log scale minimum value
        num_classes (int): Number of classes

    Returns:
    --------
        Tensor: loss
    """

    B, L, C = target.shape
    nr_mix = pred.shape[-1] // 3

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = pred[:, :, :nr_mix]
    means = pred[:, :, nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(pred[:, :, 2 * nr_mix : 3 * nr_mix], min=log_scale_min)

    centered_y = target - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - np.log((num_classes - 1) / 2)
    )
    inner_cond = (target > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (target < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)
    return -log_sum_exp(log_probs)


class BBoxOutput(object):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes = sizes
        self.translations = translations
        self.angles = angles
        self.class_labels = class_labels

    def __len__(self):
        return len(self.members)

    @property
    def members(self):
        return (self.sizes, self.translations, self.angles, self.class_labels)

    @property
    def n_classes(self):
        return self.class_labels.shape[-1]

    @property
    def device(self):
        return self.class_labels.device

    @staticmethod
    def extract_bbox_params_from_tensor(t):
        if isinstance(t, dict):
            class_labels = t["class_labels_tr"]
            translations = t["translations_tr"]
            sizes = t["sizes_tr"]
            angles = t["angles_tr"]
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]

        return class_labels, translations, sizes, angles

    @property
    def feature_dims(self):
        raise NotImplementedError()

    def get_losses(self, X_target):
        raise NotImplementedError()

    def reconstruction_loss(self, sample_params):
        raise NotImplementedError()


class AutoregressiveBBoxOutput(BBoxOutput):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes_x, self.sizes_y, self.sizes_z = sizes
        self.translations_x, self.translations_y, self.translations_z = translations
        self.class_labels = class_labels
        self.angles = angles

    @property
    def members(self):
        return (
            self.sizes_x,
            self.sizes_y,
            self.sizes_z,
            self.translations_x,
            self.translations_y,
            self.translations_z,
            self.angles,
            self.class_labels,
        )

    @property
    def feature_dims(self):
        return self.n_classes + 3 + 3 + 1

    def _targets_from_tensor(self, X_target):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_bbox_params_from_tensor(X_target)
        target = {}
        target["labels"] = target_bbox_params[0]
        target["translations_x"] = target_bbox_params[1][:, :, 0:1]
        target["translations_y"] = target_bbox_params[1][:, :, 1:2]
        target["translations_z"] = target_bbox_params[1][:, :, 2:3]
        target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
        target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
        target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
        target["angles"] = target_bbox_params[3]

        return target

    def get_losses(self, X_target):
        target = self._targets_from_tensor(X_target)

        assert torch.sum(target["labels"][..., -2]).item() == 0

        # For the class labels compute the cross entropy loss between the
        # target and the predicted labels
        label_loss = cross_entropy_loss(self.class_labels, target["labels"])

        # For the translations, sizes and angles compute the discretized
        # logistic mixture likelihood as described in
        # PIXELCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and
        # Other Modifications, by Salimans et al.
        translation_loss = dmll(self.translations_x, target["translations_x"])
        translation_loss += dmll(self.translations_y, target["translations_y"])
        translation_loss += dmll(self.translations_z, target["translations_z"])
        size_loss = dmll(self.sizes_x, target["sizes_x"])
        size_loss += dmll(self.sizes_y, target["sizes_y"])
        size_loss += dmll(self.sizes_z, target["sizes_z"])
        angle_loss = dmll(self.angles, target["angles"])

        return label_loss, translation_loss, size_loss, angle_loss

    def reconstruction_loss(self, X_target, lengths):
        # Compute the losses
        label_loss, translation_loss, size_loss, angle_loss = self.get_losses(X_target)

        label_loss = label_loss.mean()
        translation_loss = translation_loss.mean()
        size_loss = size_loss.mean()
        angle_loss = angle_loss.mean()

        return label_loss + translation_loss + size_loss + angle_loss


class Hidden2Output(nn.Module):
    def __init__(self, hidden_size, n_classes, with_extra_fc=False):
        super().__init__()
        self.with_extra_fc = with_extra_fc
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        mlp_layers = [
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
        ]
        self.hidden2output = nn.Sequential(*mlp_layers)

    def apply_linear_layers(self, x):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        class_labels = self.class_layer(x)
        translations = (self.centroid_layer_x(x), self.centroid_layer_y(x), self.centroid_layer_z(x))
        sizes = (self.size_layer_x(x), self.size_layer_y(x), self.size_layer_z(x))
        angles = self.angle_layer(x)
        return class_labels, translations, sizes, angles

    def forward(self, x, sample_params=None):
        raise NotImplementedError()


def sample_from_dmll(pred, num_classes=256):
    """Sample from mixture of logistics.

    Arguments
    ---------
        pred: NxC where C is 3*number of logistics
    """
    assert len(pred.shape) == 2

    N = pred.size(0)
    nr_mix = pred.size(1) // 3

    probs = torch.softmax(pred[:, :nr_mix], dim=-1)
    means = pred[:, nr_mix : 2 * nr_mix]
    scales = torch.nn.functional.elu(pred[:, 2 * nr_mix : 3 * nr_mix]) + 1.0001

    indices = torch.multinomial(probs, 1).squeeze()
    batch_indices = torch.arange(N, device=probs.device)
    mu = means[batch_indices, indices]
    s = scales[batch_indices, indices]
    u = torch.rand(N, device=probs.device)
    preds = mu + s * (torch.log(u) - torch.log(1 - u))

    return torch.clamp(preds, min=-1, max=1)[:, None]


class AutoregressiveDMLL(Hidden2Output):
    def __init__(self, hidden_size, n_classes, n_mixtures, bbox_output, with_extra_fc=False):
        super().__init__(hidden_size, n_classes, with_extra_fc)

        if not isinstance(n_mixtures, list):
            n_mixtures = [n_mixtures] * 7

        self.class_layer = nn.Linear(hidden_size, n_classes)

        self.fc_class_labels = nn.Linear(n_classes, 64)
        # Positional embedding for the target translation
        self.pe_trans_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_z = FixedPositionalEncoding(proj_dims=64)
        # Positional embedding for the target angle
        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        c_hidden_size = hidden_size + 64
        self.centroid_layer_x = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[0] * 3)
        self.centroid_layer_y = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[1] * 3)
        self.centroid_layer_z = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[2] * 3)
        c_hidden_size = c_hidden_size + 64 * 3
        self.angle_layer = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[6] * 3)
        c_hidden_size = c_hidden_size + 64
        self.size_layer_x = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[3] * 3)
        self.size_layer_y = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[4] * 3)
        self.size_layer_z = AutoregressiveDMLL._mlp(c_hidden_size, n_mixtures[5] * 3)

        self.bbox_output = bbox_output

    @staticmethod
    def _mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _extract_properties_from_target(sample_params):
        class_labels = sample_params["class_labels_tr"].float()
        translations = sample_params["translations_tr"].float()
        sizes = sample_params["sizes_tr"].float()
        angles = sample_params["angles_tr"].float()
        return class_labels, translations, sizes, angles

    @staticmethod
    def get_dmll_params(pred):
        assert len(pred.shape) == 2

        N = pred.size(0)
        nr_mix = pred.size(1) // 3

        probs = torch.softmax(pred[:, :nr_mix], dim=-1)
        means = pred[:, nr_mix : 2 * nr_mix]
        scales = torch.nn.functional.elu(pred[:, 2 * nr_mix : 3 * nr_mix]) + 1.0001

        return probs, means, scales

    def get_translations_dmll_params(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf).reshape(B * L, -1)
        translations_y = self.centroid_layer_y(cf).reshape(B * L, -1)
        translations_z = self.centroid_layer_z(cf).reshape(B * L, -1)

        dmll_params = {}
        p = AutoregressiveDMLL.get_dmll_params(translations_x)
        dmll_params["translations_x_probs"] = p[0]
        dmll_params["translations_x_means"] = p[1]
        dmll_params["translations_x_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_y)
        dmll_params["translations_y_probs"] = p[0]
        dmll_params["translations_y_means"] = p[1]
        dmll_params["translations_y_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_z)
        dmll_params["translations_z_probs"] = p[0]
        dmll_params["translations_z_means"] = p[1]
        dmll_params["translations_z_scales"] = p[2]

        return dmll_params

    def sample_class_labels(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape
        C = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(B * L, C)
        sampled_classes = torch.multinomial(class_probs, 1).view(B, L)
        return torch.eye(C, device=x.device)[sampled_classes]

    def sample_translations(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf)
        translations_y = self.centroid_layer_y(cf)
        translations_z = self.centroid_layer_z(cf)

        t_x = sample_from_dmll(translations_x.reshape(B * L, -1))
        t_y = sample_from_dmll(translations_y.reshape(B * L, -1))
        t_z = sample_from_dmll(translations_z.reshape(B * L, -1))
        return torch.cat([t_x, t_y, t_z], dim=-1).view(B, L, 3)

    def sample_angles(self, x, class_labels, translations):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)
        return sample_from_dmll(angles.reshape(B * L, -1)).view(B, L, 1)

    def sample_sizes(self, x, class_labels, translations, angles):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        a = self.pe_angle_z(angles)
        sf = torch.cat([tf, a], dim=-1)

        sizes_x = self.size_layer_x(sf)
        sizes_y = self.size_layer_y(sf)
        sizes_z = self.size_layer_z(sf)

        s_x = sample_from_dmll(sizes_x.reshape(B * L, -1))
        s_y = sample_from_dmll(sizes_y.reshape(B * L, -1))
        s_z = sample_from_dmll(sizes_z.reshape(B * L, -1))
        return torch.cat([s_x, s_y, s_z], dim=-1).view(B, L, 3)

    def pred_class_probs(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        b, l, _ = class_labels.shape
        c = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(b * l, c)

        return class_probs

    def pred_dmll_params_translation(self, x, class_labels):
        def dmll_params_from_pred(pred):
            assert len(pred.shape) == 2

            N = pred.size(0)
            nr_mix = pred.size(1) // 3

            probs = torch.softmax(pred[:, :nr_mix], dim=-1)
            means = pred[:, nr_mix : 2 * nr_mix]
            scales = torch.nn.functional.elu(pred[:, 2 * nr_mix : 3 * nr_mix])
            scales = scales + 1.0001

            return probs, means, scales

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        t_x = self.centroid_layer_x(cf).reshape(B * L, -1)
        t_y = self.centroid_layer_y(cf).reshape(B * L, -1)
        t_z = self.centroid_layer_z(cf).reshape(B * L, -1)

        return dmll_params_from_pred(t_x), dmll_params_from_pred(t_y), dmll_params_from_pred(t_z)

    def forward(self, x, sample_params):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        # Extract the target properties from sample_params and embed them into
        # a higher dimensional space.
        target_properties = AutoregressiveDMLL._extract_properties_from_target(sample_params)

        class_labels = target_properties[0]
        translations = target_properties[1]
        angles = target_properties[3]

        c = self.fc_class_labels(class_labels)

        tx = self.pe_trans_x(translations[:, :, 0:1])
        ty = self.pe_trans_y(translations[:, :, 1:2])
        tz = self.pe_trans_z(translations[:, :, 2:3])

        a = self.pe_angle_z(angles)
        class_labels = self.class_layer(x)

        cf = torch.cat([x, c], dim=-1)
        # Using the true class label we now want to predict the translations
        translations = (self.centroid_layer_x(cf), self.centroid_layer_y(cf), self.centroid_layer_z(cf))
        tf = torch.cat([cf, tx, ty, tz], dim=-1)
        angles = self.angle_layer(tf)
        sf = torch.cat([tf, a], dim=-1)
        sizes = (self.size_layer_x(sf), self.size_layer_y(sf), self.size_layer_z(sf))

        return self.bbox_output(sizes, translations, angles, class_labels)


class FrozenBatchNorm2d(nn.Module):
    """A BatchNorm2d wrapper for Pytorch's BatchNorm2d where the batch
    statictis are fixed.
    """

    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.register_parameter("weight", Parameter(torch.ones(num_features)))
        self.register_parameter("bias", Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def extra_repr(self):
        return "{num_features}".format(**self.__dict__)

    @classmethod
    def from_batch_norm(cls, bn):
        fbn = cls(bn.num_features)
        # Update the weight and biases based on the corresponding weights and
        # biases of the pre-trained bn layer
        with torch.no_grad():
            fbn.weight[...] = bn.weight
            fbn.bias[...] = bn.bias
            fbn.running_mean[...] = bn.running_mean
            fbn.running_var[...] = bn.running_var + bn.eps
        return fbn

    @staticmethod
    def _getattr_nested(m, module_names):
        if len(module_names) == 1:
            return getattr(m, module_names[0])
        else:
            return FrozenBatchNorm2d._getattr_nested(getattr(m, module_names[0]), module_names[1:])

    @staticmethod
    def freeze(m):
        for name, layer in m.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                nest = name.split(".")
                if len(nest) == 1:
                    setattr(m, name, FrozenBatchNorm2d.from_batch_norm(layer))
                else:
                    setattr(
                        FrozenBatchNorm2d._getattr_nested(m, nest[:-1]),
                        nest[-1],
                        FrozenBatchNorm2d.from_batch_norm(layer),
                    )

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks."""

    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, X):
        return self._feature_extractor(X)


class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs.
    """

    def __init__(self, freeze_bn, input_channels, feature_size):
        super(ResNet18, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.resnet18(weights=None)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self._feature_extractor.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.feature_size))
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims // 2
        exb = 2 * torch.linspace(0, ll - 1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1)
        self.sigma = 2 * torch.pi * self.sigma

    def forward(self, x):
        return torch.cat([torch.sin(x * self.sigma.to(x.device)), torch.cos(x * self.sigma.to(x.device))], dim=-1)


class BaseAutoregressiveTransformer(nn.Module):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__()
        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get("feed_forward_dimensions", 3072),
            attention_type="full",
            activation="gelu",
        ).get()

        self.register_parameter("start_token_embedding", nn.Parameter(torch.randn(1, 512)))

        # TODO: Add the projection dimensions for the room features in the
        # config!!!
        self.feature_extractor = feature_extractor
        self.fc_room_f = nn.Linear(self.feature_extractor.feature_size, 512)

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3 to account for the masked bins for the size, position and
        # angle properties
        self.input_dims = input_dims
        self.n_classes = self.input_dims - 3 - 3 - 1
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)

        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(512, hidden_dims)
        self.hidden2output = hidden2output

    def start_symbol(self, device="cpu"):
        start_class = torch.zeros(1, 1, self.n_classes, device=device)
        start_class[0, 0, -2] = 1
        return {
            "class_labels": start_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device),
        }

        return boxes

    def end_symbol(self, device="cpu"):
        end_class = torch.zeros(1, 1, self.n_classes, device=device)
        end_class[0, 0, -1] = 1
        return {
            "class_labels": end_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device),
        }

    def start_symbol_features(self, B, room_mask):
        room_layout_f = self.fc_room_f(self.feature_extractor(room_mask))
        return room_layout_f[:, None, :]

    def forward(self, sample_params):
        raise NotImplementedError()

    def autoregressive_decode(self, boxes, room_mask):
        raise NotImplementedError()

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        raise NotImplementedError()


class AutoregressiveTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter("empty_token_embedding", nn.Parameter(torch.randn(1, 512)))

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, _, _ = class_labels.shape

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(sample_params["lengths"] + 2, max_len=X.shape[1])
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, _, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_y(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_z(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_y(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_z(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    def autoregressive_decode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        # Sample the class label for the next bbbox
        class_labels = self.hidden2output.sample_class_labels(F)
        # Sample the translations
        translations = self.hidden2output.sample_translations(F, class_labels)
        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_labels, translations)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(F, class_labels, translations, angles)

        return {"class_labels": class_labels, "translations": translations, "sizes": sizes, "angles": angles}

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        """
        Generates scene given room mask, generating coarse masks for each object.

        :param room_mask: float tensor representing binary room mask, with 1s in valid positions and 0s otherwise
        :param max_boxes: maximum number of objects to generate, defaults to 32
        :param device: device on which to move tensors and perform computation, defaults to "cpu"
        :return: dict of tensors in the form
            {
                "class_labels": tensor of class labels of objects,
                "translations": tensor of positions of objects in scene,
                "sizes": tensor of approximate sizes of objects in scene,
                "angles": tensor of rotation matrix to apply to each object in scene
            }
        """
        room_mask.to(device)
        boxes = self.start_symbol(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu"),
        }

    def autoregressive_decode_with_class_label(self, boxes, room_mask, class_label):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the translations conditioned on the query_class_label
        translations = self.hidden2output.sample_translations(F, class_label)
        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translations)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(F, class_label, translations, angles)

        return {"class_labels": class_label, "translations": translations, "sizes": sizes, "angles": angles}

    @torch.no_grad()
    def add_object(self, room_mask, class_label, boxes=None, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label(boxes=boxes, room_mask=room_mask, class_label=class_label)

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"],
        }

    @torch.no_grad()
    def complete_scene(self, boxes, room_mask, max_boxes=100, device="cpu"):
        boxes = dict(boxes.items())

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"],
        }

    def autoregressive_decode_with_class_label_and_translation(self, boxes, room_mask, class_label, translation):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(F, class_label, translation, angles)

        return {"class_labels": class_label, "translations": translation, "sizes": sizes, "angles": angles}

    @torch.no_grad()
    def add_object_with_class_and_translation(self, boxes, room_mask, class_label, translation, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes, class_label=class_label, translation=translation, room_mask=room_mask
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"],
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(self, boxes, room_mask, class_label, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(F, class_label)


def atiss_network(config):
    n_classes = len(config["data"]["classes"])
    feature_size = 7 + n_classes  #  translation (3) + size (3) + rotation (1) + one-hot-class-encoding
    hidden2output_layer = AutoregressiveDMLL(
        config["network"].get("hidden_dims", 768),
        n_classes,
        config["network"].get("n_mixtures", 4),
        AutoregressiveBBoxOutput,
        config["network"].get("with_extra_fc", False),
    )
    feature_extractor = ResNet18(
        freeze_bn=config["feature_extractor"].get("freeze_bn", True),
        input_channels=config["feature_extractor"].get("input_channels", 1),
        feature_size=config["feature_extractor"].get("feature_size", 256),
    )
    network = AutoregressiveTransformer(feature_size, hidden2output_layer, feature_extractor, config["network"])
    # Check whether there is a weight file provided to continue training from
    weight_file = config.get("weight_file", None)
    device = config.get("device", "cpu")
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        network.load_state_dict(torch.load(weight_file, map_location=device))
    network.to(device)

    return network


# dataset-related


def descale(x, minimum, maximum):
    x = (x + 1) / 2
    x = x * (maximum - minimum) + minimum
    return x


def descale_bbox_params(bounds, s):
    sample_params = {}
    for k, v in s.items():
        if k == "room_layout" or k == "class_labels" or k == "objfeats":
            sample_params[k] = v
        else:
            sample_params[k] = descale(v, np.asarray(bounds[k][0]), np.asarray(bounds[k][1]))
    return sample_params
