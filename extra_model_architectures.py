import torch
from torch import nn


class BRDF_normal_predictor_with_incoming_light(nn.Module):
    """
    Version of the normal predictor that also takes in the incoming light for a pixel being queried.
    Currently an archive, and unused.
    """

    def __init__(
        self, img_height: int, img_width: int, incoming_light_size: int
    ) -> None:
        super().__init__()

        # Take in an RGB-D image and run multiple conv-net layers on it
        # TODO: Add dropouts, pooling, etc.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=28, kernel_size=3, padding="same"),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=4, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        # TODO: Define basic architecture

        self.img_height = img_height
        self.img_width = img_width
        self.incoming_light_size = incoming_light_size

        self.diffuse_brdf_size = 3
        self.spec_brdf_size = 4
        self.normal_size = 3
        self.max_spec_c = 16

        # Layer Norm before output
        self.norm1 = nn.LayerNorm(
            self.img_height * self.img_width * 4 + self.incoming_light_size
        )

        # One network for the BRDFs, another for the normals.
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=(self.img_height * self.img_width * 4)
                + self.incoming_light_size,
                out_features=self.diffuse_brdf_size + self.spec_brdf_size,
            ),
            # nn.Sigmoid(),
            # nn.Softplus(beta=10),
            # nn.LeakyReLU(0.01),
            # nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=(self.img_height * self.img_width * 4)
                + self.incoming_light_size,
                out_features=self.normal_size,
            ),
            nn.Tanh(),
            # TODO: TanH Activation?
        )
        # TODO: support multiple types of BRDFs eventually?

    def forward(
        self, image: torch.Tensor, incoming_light: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Input:
            image: (N, C, H, W)
            incoming_light: (N, R, C) where each index of N is the incoming light for one point (R rays, C colors per ray)
            TODO: support incoming light for multiple points?

        """

        N = image.size(0)

        img_features = self.conv1(image)
        img_features = self.conv2(img_features)

        img_features = img_features.reshape(N, -1)  # (N, C * H * W)

        # Concatenate incoming light to be used for MLP
        # There's only one pixel so for now this is how it's being used,
        # But should be explored how to improve this.
        # TODO: Transform incoming light in some way?
        img_and_light_features = torch.cat(
            [img_features, incoming_light.reshape(N, -1)], dim=1
        )

        # TODO: Use the norm?
        # print(f"{img_and_light_features = }")
        # img_and_light_features = self.norm1(img_and_light_features)

        brdf_predictions = self.fc1(img_and_light_features)
        normal_predictions = self.fc2(img_and_light_features)

        # "Soft Capping" Trick: https://pytorch.org/blog/flexattention/
        # TODO: Remove softplus for this exp? We want it to be positive so I don't mind keeping it.
        spec_c = brdf_predictions[:, -1]
        spec_c = spec_c / self.max_spec_c
        spec_c = nn.functional.tanh(spec_c)
        spec_c = spec_c * self.max_spec_c

        # TODO: refine arguments
        # Best way to structure this...all as one tensor or as multiple?
        output_dict = {
            "brdf": {
                "diffuse": brdf_predictions[:, : self.diffuse_brdf_size],
                # RGB + specular C
                "specular": brdf_predictions[
                    :,
                    self.diffuse_brdf_size : self.diffuse_brdf_size
                    + self.spec_brdf_size
                    - 1,
                ],
                "specular_c": spec_c,
            },
            "normal": normal_predictions,
        }

        return output_dict
