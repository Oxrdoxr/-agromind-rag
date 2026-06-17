# src/image_retriever.py

from pathlib import Path
from typing import Dict, Any, List

import chromadb

from src.config import config
from src.image_embeddings import clip_embeddings


class ImageRetriever:
    """
    Retrieve similar crop disease images from ChromaDB.
    """

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=config.chromadb_path
        )

        try:

            self.collection = (
                self.client.get_collection(
                    config.retrieval.image_collection
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"Image collection not found: "
                f"{config.retrieval.image_collection}"
            ) from e

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search_by_image(
        self,
        image_path: str,
        k: int = 5
    ) -> Dict[str, Any]:

        image_path = Path(image_path)

        if not image_path.exists():

            return {
            "success": True,
            "matches": len(results["ids"][0]),
            "results": results
        }

        embedding = (
            clip_embeddings
            .embed_image(image_path)
        )

        results = self.collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return {
            "success": True,
            "results": results
        }

    # --------------------------------------------------
    # Diagnose
    # --------------------------------------------------

    def diagnose(
        self,
        image_path: str,
        k: int = 5
    ) -> Dict[str, Any]:

        search = self.search_by_image(
            image_path=image_path,
            k=k
        )

        if not search["success"]:

            return search

        results = search["results"]

        if (
            not results["ids"]
            or not results["ids"][0]
        ):

            return {
                "success": False,
                "message":
                    "No similar images found."
            }

        best_metadata = (
            results["metadatas"][0][0]
        )

        best_id = (
            results["ids"][0][0]
        )

        best_distance = (
            results["distances"][0][0]
        )

        similarity_score = max(
            0.0,
            min(
                1.0,
                1 - best_distance
            )
        )

        top_matches = []

        for i in range(
            len(results["ids"][0])
        ):

            distance = (
                results["distances"][0][i]
            )

            score = max(
                0.0,
                min(
                    1.0,
                    1 - distance
                )
            )

            metadata = (
                results["metadatas"][0][i]
            )

            top_matches.append({

                "image_id":
                    results["ids"][0][i],

                "crop":
                    metadata.get("crop"),

                "disease":
                    metadata.get("disease"),

                "disease_type":
                    metadata.get(
                        "disease_type"
                    ),

                "confidence":
                    round(score, 4)
            })

        return {

            "success": True,

            "image_id":
                best_id,

            "crop":
                best_metadata.get("crop"),

            "disease":
                best_metadata.get("disease"),

            "disease_type":
                best_metadata.get(
                    "disease_type"
                ),

            "confidence":
                round(similarity_score, 4),

            "matches":
                len(results["ids"][0]),

            "top_matches":
                top_matches
        }


if __name__ == "__main__":

    retriever = ImageRetriever()

    result = retriever.diagnose(
        image_path=(
            "data/annotated_images/images/"
            "b24cb2c9_crop01.jpg"
        ),
        k=5
    )

    print("\nRESULT")
    print("=" * 50)
    print(result)