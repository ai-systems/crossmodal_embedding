import ml_metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from loguru import logger


def single_retrieval_recall(retrieved, relevant):
    relevant_retrieved = list(set(retrieved) & set(relevant))
    return len(relevant_retrieved) / len(relevant)


def compute_map_with_unification(premises, embeddings, is_premise_of, top_k_value=30):

    real_values = list()
    retrieved_values = list()

    for title, premise in tqdm(premises.items(), desc="Computing MAP with unification"):
        sim = dict()
        new_relevance_score = dict()
        if title in embeddings:
            embedding_keys = list(embeddings.keys())
            embedding_values = list(embeddings.values())
            if title in embeddings:
                similarity_results = cosine_similarity(
                    [embeddings[title]], embedding_values
                )[0]
                for i in range(len(embedding_values)):
                    sim[embedding_keys[i]] = similarity_results[i]
            sim = {
                k: v
                for k, v in sorted(sim.items(), key=lambda item: item[1], reverse=True)
            }
            top_k_sim = list(sim.keys())[1 : top_k_value + 1]
            unification_score = dict()
            new_relevance_score = dict()
            for title_e, _ in sim.items():
                unification_score[title_e] = 0
                if title_e in is_premise_of:
                    for p in is_premise_of[title_e]:
                        if p in top_k_sim:

                            unification_score[title_e] = (
                                unification_score[title_e] + sim[p]
                            )

                new_relevance_score[title_e] = unification_score[title_e] + sim[title_e]
                # new_relevance_score[title_e] = unification_score[title_e]

            new_relevance_score = {
                k: v
                for k, v in sorted(
                    new_relevance_score.items(), key=lambda item: item[1], reverse=True
                )
            }

            contained_premises = list()
            for p in premise:
                if p in sim:
                    contained_premises.append(p)

            if contained_premises:
                real_values.append(contained_premises)
                retrieved_values.append(list(new_relevance_score.keys())[1:])

                # logger.info(f"Title: {title}")
                # logger.info(f"Real: {contained_premises}")
                # for p in contained_premises:
                #     logger.info(f"Real similarity: {new_relevance_score[p]}")
                # logger.info(f"Obtained top k: {top_k_sim}")
                # logger.info(f"Obtained later: {list(new_relevance_score.keys())[1:10]}")
                # logger.info(
                #     f"Individual map: {metrics.mapk([contained_premises], [list(new_relevance_score.keys())], len(sim)) }"
                # )
                # input()
    map_value = metrics.mapk(real_values, retrieved_values, 50)

    return map_value


def compute_map_basic(premises, embeddings):

    real_values = list()
    retrieved_values = list()
    for title, premise in tqdm(premises.items(), desc="Computing MAP"):
        sim = dict()
        embedding_keys = list(embeddings.keys())
        embedding_values = list(embeddings.values())
        if title in embeddings:
            similarity_results = cosine_similarity(
                [embeddings[title]], embedding_values
            )[0]
            for i in range(len(embedding_values)):
                sim[embedding_keys[i]] = similarity_results[i]

            sim = {
                k: v
                for k, v in sorted(sim.items(), key=lambda item: item[1], reverse=True)
            }

            contained_premises = list()
            for p in premise:
                if p in sim:
                    contained_premises.append(p)

            if contained_premises:
                real_values.append(contained_premises)
                retrieved_values.append(list(sim.keys())[1:])

                # logger.info(f"Title: {title}")
                # logger.info(f"Real: {contained_premises}")
                # for p in contained_premises:
                #     logger.info(f"Real similarity: {sim[p]}")
                # logger.info(f"Obtained: {list(sim.keys())[1:10]}")
                # logger.info(f"Obtained values: {list(sim.values())[1:10]}")
                # logger.info(
                #     f"Individual map: {metrics.mapk([contained_premises], [list(sim.keys())], len(sim)) }"
                # )
                # input()
    map_value = metrics.mapk(real_values, retrieved_values, 50)

    return map_value
