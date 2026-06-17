"""
Image embedding module using CLIP for multimodal search.
All settings come from config.yaml – no hardcoded paths or models.


Responsibilities:

Load CLIP once
Embed images
Embed text
Normalize vectors
"""
from pathlib import Path
from typing import List, Union

import clip
import numpy as np
import torch
from PIL import Image
from langchain_core.embeddings import Embeddings

from src.config import config


class CLIPEmbeddings(Embeddings):
    """
    Singleton CLIP embedding wrapper.

    Config:

    image:
      model: "ViT-B/32"
      normalize: true
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_name = config.image.model

        self.normalize = getattr(
            config.image,
            "normalize",
            True
        )

        print(
            f"🚀 Loading CLIP '{self.model_name}' "
            f"on {self.device}"
        )

        self.model, self.preprocess = clip.load(
            self.model_name,
            device=self.device
        )

        self.model.eval()

        print(
            f"✅ CLIP loaded "
            f"(dimension={self.model.visual.output_dim})"
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _normalize_vector(
        self,
        vector: np.ndarray
    ) -> np.ndarray:

        if not self.normalize:
            return vector

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector

        return vector / norm

    # --------------------------------------------------
    # Image Embeddings
    # --------------------------------------------------

    def embed_image(
        self,
        image_path: Union[str, Path]
    ) -> np.ndarray:

        image_path = Path(image_path)

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        image_tensor = (
            self.preprocess(image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            embedding = (
                self.model.encode_image(
                    image_tensor
                )
            )

            embedding = (
                embedding /
                embedding.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        embedding = (
            embedding.cpu()
            .numpy()
            .astype(np.float32)
            .flatten()
        )

        return self._normalize_vector(
            embedding
        )

    def embed_images(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[np.ndarray]:

        return [
            self.embed_image(path)
            for path in image_paths
        ]

    # --------------------------------------------------
    # Text Embeddings
    # --------------------------------------------------

    def embed_text(
        self,
        text: str
    ) -> np.ndarray:

        text_tensor = (
            clip.tokenize([text])
            .to(self.device)
        )

        with torch.no_grad():

            embedding = (
                self.model.encode_text(
                    text_tensor
                )
            )

            embedding = (
                embedding /
                embedding.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        embedding = (
            embedding.cpu()
            .numpy()
            .astype(np.float32)
            .flatten()
        )

        return self._normalize_vector(
            embedding
        )

    # --------------------------------------------------
    # LangChain Compatibility
    # --------------------------------------------------

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        return [
            self.embed_text(text).tolist()
            for text in texts
        ]

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        return self.embed_text(
            text
        ).tolist()

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_dimension(self) -> int:

        return self.model.visual.output_dim

    def cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:

        emb1 = self._normalize_vector(emb1)
        emb2 = self._normalize_vector(emb2)

        return float(
            np.dot(emb1, emb2)
        )


# Singleton instance
clip_embeddings = CLIPEmbeddings()


if __name__ == "__main__":

    print("=" * 60)
    print("CLIP EMBEDDING TEST")
    print("=" * 60)

    print(
        f"Model: {clip_embeddings.model_name}"
    )

    print(
        f"Dimension: "
        f"{clip_embeddings.get_dimension()}"
    )

    text_embedding = (
        clip_embeddings.embed_text(
            "Tomato Early Blight"
        )
    )

    print(
        f"Text shape: "
        f"{text_embedding.shape}"
    )

    print(
        f"Vector norm: "
        f"{np.linalg.norm(text_embedding):.4f}"
    )

    print("✅ Ready")