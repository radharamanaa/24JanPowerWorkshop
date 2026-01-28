from agent_framework import ai_function
from qdrant_client.http.models import QueryResponse

from utils import get_transformer_model


def get_prepared_collection(collection_name:str):
    @ai_function
    def search_in_qdrant(query:str):
        """
        Searches the qdrant collection for the query and returns the top 5 results.
        :param query: Query text you want similar text for.
        :return: Top 5 results.
        """
        print(f"searching in qdrant for query: {query}")
        from utils import qdrant_client
        results:QueryResponse = qdrant_client.query_points(collection_name=collection_name,
                                             query=get_transformer_model().encode(query), limit=5)

        results_ = [f"Score: {result.score}, Text: {result.payload["text"]}" for result in results.points]
        for item in results_:
            print(item)
        return results_
    return search_in_qdrant