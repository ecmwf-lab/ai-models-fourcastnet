# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import torch
from ai_models.model import Model

from .afnonet import AFNONet, PrecipNet, unlog_tp_torch  # noqa

LOG = logging.getLogger(__name__)


class FourCastNet(Model):
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.0/{file}"
    )
    download_files = [
        "backbone.ckpt",
        "precip.ckpt",
        "global_means.npy",
        "global_stds.npy",
    ]

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    # Output
    expver = "fcnt"

    def __init__(self, precip_flag=True, **kwargs):
        super().__init__(**kwargs)

        self.precip_flag = precip_flag
        self.n_lat = 720
        self.n_lon = 1440
        self.hour_steps = 6
        self.precip_channels = 20

        self.backbone_channels = len(self.ordering)

    def load_statistics(self):
        path = os.path.join(self.assets, "global_means.npy")
        LOG.info("Loading %s", path)
        self.means = np.load(path)
        self.means = self.means[:, : self.backbone_channels, ...]
        self.means = self.means.astype(np.float32)

        path = os.path.join(self.assets, "global_stds.npy")
        LOG.info("Loading %s", path)
        self.stds = np.load(path)
        self.stds = self.stds[:, : self.backbone_channels, ...]
        self.stds = self.stds.astype(np.float32)

    def load_model(self, checkpoint_file, precip=False):
        out_channels = 1 if precip else self.backbone_channels
        in_channels = 20 if precip else self.backbone_channels

        model = AFNONet(in_chans=in_channels, out_chans=out_channels)

        if precip:
            model = PrecipNet(backbone=model)

        model.zero_grad()
        # Load weights

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        asset_dim = checkpoint["model_state"][
            tuple(checkpoint["model_state"])[1]
        ].shape[1]
        model_dim = self.precip_channels if precip else self.backbone_channels

        if asset_dim != model_dim:
            raise ValueError(
                f"Asset version ({asset_dim} variables) does not match model version"
                f"({model_dim} variables), please redownload correct weights."
            )

        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])
        # Set model to eval mode and return
        model.eval()
        model.to(self.device)
        return model

    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if reverse:
            new_data = data * self.stds + self.means
        else:
            new_data = (data - self.means) / self.stds
        return new_data

    unlog_tp_torch = unlog_tp_torch

    def copy_extend(self, data):
        return np.concatenate((data, data[:, :, [-1], :]), axis=2)

    def nan_extend(self, data):
        return np.concatenate(
            (data, np.full_like(data[:, :, [-1], :], np.nan, dtype=data.dtype)), axis=2
        )

    def run(self):
        self.load_statistics()

        all_fields = self.all_fields
        all_fields = all_fields.sel(
            param_level=self.ordering, remapping={"param_level": "{param}{levelist}"}
        )
        all_fields = all_fields.order_by(
            {"param_level": self.ordering},
            remapping={"param_level": "{param}{levelist}"},
        )

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Field ordering:")
            for i, (field, name) in enumerate(zip(all_fields, self.ordering)):
                LOG.debug("Field %d %r %s", i, field, name)

        all_fields_numpy = all_fields.to_numpy(dtype=np.float32)[np.newaxis, :, :-1, :]

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Mean/stdev:")

            for i, name in enumerate(self.ordering):
                LOG.debug(
                    "    %s %s %s %s %s %s",
                    name,
                    self.means[:, i].flatten(),
                    self.stds[:, i].flatten(),
                    np.mean(all_fields_numpy[:, i]),
                    np.std(all_fields_numpy[:, i]),
                    abs(np.mean(all_fields_numpy[:, i]) - self.means[:, i].flatten())
                    / max(
                        abs(self.means[:, i].flatten()),
                        abs(np.mean(all_fields_numpy[:, i])),
                    ),
                )

        all_fields_numpy = self.normalise(all_fields_numpy)

        backbone_ckpt = os.path.join(self.assets, "backbone.ckpt")
        with self.timer(f"Loading {backbone_ckpt}"):
            backbone_model = self.load_model(backbone_ckpt)

        if self.precip_flag:
            precip_ckpt = os.path.join(self.assets, "precip.ckpt")
            with self.timer(f"Loading {precip_ckpt}"):
                precip_model = self.load_model(precip_ckpt, precip=True)

        # Run the inference session
        input_iter = torch.from_numpy(all_fields_numpy).to(self.device)
        self.hour_steps = 6

        if self.precip_flag:
            self.accumulate = torch.zeros((1, 1, self.n_lat, self.n_lon))

        sample_sfc = all_fields.sel(param="2t")[0]

        if self.precip_flag:
            self.write_input_fields(all_fields, ["tp"], sample_sfc)
        else:
            self.write_input_fields(all_fields)

        torch.set_grad_enabled(False)

        with self.stepper(6) as stepper:
            for i in range(self.lead_time // self.hour_steps):
                output = backbone_model(input_iter)
                if self.precip_flag:
                    precip_output = precip_model(output[:, : self.precip_channels, ...])
                    self.accumulate += unlog_tp_torch(precip_output.cpu())

                input_iter = output

                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of normalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s %s %s %s %s",
                            name,
                            np.mean(output[:, j].cpu().numpy()),
                            np.std(output[:, j].cpu().numpy()),
                            np.amin(output[:, j].cpu().numpy()),
                            np.amax(output[:, j].cpu().numpy()),
                        )

                # Save the results
                step = (i + 1) * 6
                output = self.nan_extend(
                    self.normalise(output.cpu().numpy(), reverse=True)
                )

                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of denormalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s mean=%s std=%s min=%s max=%s %s %s",
                            name,
                            np.mean(output[:, j]),
                            np.std(output[:, j]),
                            np.amin(output[:, j]),
                            np.amax(output[:, j]),
                            self.means[:, j].flatten(),
                            self.stds[:, j].flatten(),
                        )

                if self.precip_flag:
                    precip_output = self.nan_extend(self.accumulate.numpy())

                for k, fs in enumerate(all_fields):
                    self.write(
                        output[0, k, ...],
                        check_nans=True,
                        template=fs,
                        step=step,
                    )

                if self.precip_flag:
                    self.write(
                        precip_output.squeeze(),
                        check_nans=True,
                        template=sample_sfc,
                        step=step,
                        param="tp",
                        stepType="accum",
                    )

                stepper(i, step)


class FourCastNet0(FourCastNet):
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.0/{file}"
    )

    assets_extra_dir = "0.0"

    param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv"]

    param_level_pl = (["t", "u", "v", "z", "r"], [1000, 850, 500, 50])

    ordering = [
        "10u",
        "10v",
        "2t",
        "sp",
        "msl",
        "t850",
        "u1000",
        "v1000",
        "z1000",
        "u850",
        "v850",
        "z850",
        "u500",
        "v500",
        "z500",
        "t500",
        "z50",
        "r500",
        "r850",
        "tcwv",
    ]


class FourCastNet1(FourCastNet):
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.1/{file}"
    )

    param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

    param_level_pl = (["t", "u", "v", "z", "r"], [1000, 850, 500, 250, 50])

    assets_extra_dir = "0.1"

    ordering = [
        "10u",
        "10v",
        "2t",
        "sp",
        "msl",
        "t850",
        "u1000",
        "v1000",
        "z1000",
        "u850",
        "v850",
        "z850",
        "u500",
        "v500",
        "z500",
        "t500",
        "z50",
        "r500",
        "r850",
        "tcwv",
        "100u",
        "100v",
        "u250",
        "v250",
        "z250",
        "t250",
    ]


def model(model_version, **kwargs):
    models = {
        "0": FourCastNet0,
        "1": FourCastNet1,
        "release": FourCastNet0,
        "latest": FourCastNet1,
    }
    return models[model_version](**kwargs)
