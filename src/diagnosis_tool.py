from src.image_retriever import ImageRetriever
from src.retriever import retrieval_tool

image_retriever = ImageRetriever()

def diagnose_crop_image(
    image_path: str,
    user_text: str | None = None
):

    diagnosis = image_retriever.diagnose(
        image_path=image_path,
        k=5
    )

    if not diagnosis["success"]:

        return {
            "success": False,
            "message": diagnosis.get(
                "message",
                "Diagnosis failed."
            )
        }

    disease = diagnosis.get(
        "disease",
        ""
    )

    crop = diagnosis.get(
        "crop",
        ""
    )

    disease_type = diagnosis.get(
        "disease_type",
        ""
    )

    confidence = diagnosis.get(
        "confidence",
        0.0
    )

    products = retrieval_tool.search_products(
        f"{crop} {disease}",
        k=5
    )

    support_cases = retrieval_tool.search_support_cases(
        query=disease,
        k=3,
        category="diagnosis"
    )

    return {

        "success": True,

        "diagnosis": {

            "crop": crop,

            "disease": disease,

            "disease_type": disease_type,

            "confidence": confidence
        },

        "recommended_products":
            products,

        "historical_cases":
            support_cases,

        "similar_images":
            diagnosis.get(
                "top_matches",
                []
            )
    }