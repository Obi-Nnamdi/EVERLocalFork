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


# TODO: Could be turned into a more general BRDF class when implementing more complex models
# TODO: Might be worth looking into the dw_i term in the rendering equation and maybe thinking of just multipling a sort of area patch term?
class Blinn_Phong_BRDF:
    """
    Implemented from here: https://rodolphe-vaillant.fr/entry/85/phong-illumination-model-cheat-sheet

    NOTE: Currently Unused.
    """

    def __init__(
        self, Kd: torch.Tensor, Ks: torch.Tensor, spec_reflect_c: torch.Tensor
    ) -> None:
        """
        Initialize a Blinn Phong BRDF with Diffuse, Specular, and specular reflection size values.
        Inputs:
            Kd: (P, 3)
            Ks: (P, 3)
            spec_reflect_c: (P,)
        """

        P = Kd.size(0)
        assert Kd.size(1) == 3
        assert Ks.size(1) == 3
        assert spec_reflect_c.size(0) == P

        self.Kd = Kd  # (P, 3)
        self.Ks = Ks  # (P, 3)
        # Control spectular reflection size
        self.spec_reflect_c = spec_reflect_c  # (P,)

    def compute_diffuse(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
        Outputs:
            diffuse_color: (N, P, 3) R,G,B values of diffuse lighting contributions at each of P points for all of the N directions
        """

        # Compute dot products of normal and all light directions
        diffuse_terms = incoming_light_dirs @ normal.T  # (N, 3) @ (3, P) --> (N, P)
        diffuse_terms = diffuse_terms.unsqueeze(-1)  # (N, P, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light
        N = incoming_light.size(0)
        P = normal.size(0)
        expanded_kd = self.Kd.unsqueeze(0).expand(N, -1, -1)  # (N (new), P, 3)
        expanded_incoming_light = incoming_light.unsqueeze(1).expand(
            -1, P, -1
        )  # (N, P (new), 3)

        diffuse_intensity = expanded_kd * expanded_incoming_light  # (N, P, 3)

        # Final Diffuse colors for each light source
        return diffuse_terms * diffuse_intensity  # (N, P, 3)

    def compute_specular(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        normal: torch.Tensor,
        outgoing_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Uses Blinn model of specular reflection.

        incoming_light: (N, 3) R,G,B values of incoming light for each direction
        incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space
        normal: (P, 3) Surface normal in world space
        outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)

        Outputs:
            specular_color: (N, P, 3) R,G,B values of specular lighting contributions at each of P points for all of the N directions
        """

        N = incoming_light.size(0)
        P = normal.size(0)

        # Compute specular term
        halfway_vecs = torch.nn.functional.normalize(
            (incoming_light_dirs + outgoing_dir)
        )  # (N, 3)

        # Compute dot products of normal and all halfway directions
        specular_term_intermediate = (
            halfway_vecs @ normal.T
        )  # (N, 3) @ (3, P) --> (N, P)
        specular_term = torch.pow(
            specular_term_intermediate,
            self.spec_reflect_c.unsqueeze(0).expand(N, -1),
        )  # (N, P), [specular exp reshaped to (N, P)]
        specular_term = specular_term.unsqueeze(-1)  # (N, P, 1)

        # TODO: Don't want negative dot product values (light directions are spherical), but I'm choosing to not clamp for now
        # (could uncomment this line, but I assume everything passed in has a positive dot product)
        # light_intensity_dot_products = torch.clamp(light_intensity_dot_products, min = 0)

        # Combine diffuse color of material with the incoming light

        specular_intensity = self.Ks.unsqueeze(0).expand(
            N, -1, -1
        ) * incoming_light.unsqueeze(1).expand(
            -1, P, -1
        )  # (N (new), P, 3) * (N, P (new), 3)

        # Final colors for each light source
        return specular_term * specular_intensity  # (N, P, 1) * (N, P, 3) => (N, P, 3)

    # TODO: Construct as a general function
    def construct_outgoing_radiance(
        self,
        incoming_light: torch.Tensor,
        incoming_light_dirs: torch.Tensor,
        outgoing_dir: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            incoming_light: (P, N, 3) R,G,B values of incoming light for each direction
            incoming_light_dirs: (P, N, 3) Directions oriented towards the light source in world space
            normal: (P, 3) Surface normals in world space for each of P points
            outgoing_dir: (1, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point)

        Outputs:



        Returns color as a (P, 3) tensor for every point.

        Sources:

        https://en.wikipedia.org/wiki/Rendering_equation
        https://15462.courses.cs.cmu.edu/spring2024content/lectures/15_brdfs/15_brdfs_slides.pdf
        https://www.scratchapixel.com/lessons/3d-basic-rendering/phong-shader-BRDF/phong-illumination-models-brdf.html
        """

        diffuse_component = self.compute_diffuse(
            incoming_light, incoming_light_dirs, normals
        )
        specular_component = self.compute_specular(
            incoming_light,
            incoming_light_dirs,
            normals,
            outgoing_dir,
        )

        # TODO: Proper intergration of diffuse and specular terms across all the lights? Should I just average?
        # Should I take weighted average w/ something else? (i.e. normal weighting)

        # Simple Diffuse + Specular combination
        full_lighting = diffuse_component + specular_component  # (N, P, 3)

        # Compute light attenuation based on viewing direction as found in rendering equation
        # <w_i, n>
        light_weakening_factors = (
            incoming_light_dirs @ normals.T
        )  # (N, 3) @ (3, P) --> (N, P)
        light_weakening_factors = light_weakening_factors.unsqueeze(-1)  # (N, P, 1)

        # Only choose light sources that are on the same side as the surface normal.
        positive_light_contributions = light_weakening_factors >= 0  # (N, P, 1)

        # We now only keep the positive lighting contributions (same side hemisphere) out of all the lighting contributions
        positive_masked_lighting = full_lighting.masked_fill(
            ~positive_light_contributions, 0.0
        )  # (N, P, 3)

        # Manually take the mean across masked elements
        return torch.sum(positive_masked_lighting, dim=0) / torch.sum(
            positive_light_contributions, dim=0
        )  # (P, 3)
