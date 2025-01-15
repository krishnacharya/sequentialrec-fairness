import pandas as pd

def get_irmeasures_qrels(test_data):
    qrels = test_data.rename(columns = {"user_id": "query_id", "item_id": "doc_id"}) 
    qrels["relevance"] = 1
    qrels.query_id = qrels.query_id.astype("str")
    qrels.doc_id = qrels.doc_id.astype("str")
    return qrels[["query_id", "doc_id", "relevance"]]

def get_irmeasures_run(recs, test_data) -> pd.DataFrame:
    all_rows = [] 
    for user_id, user_recs in zip(test_data.user_id, recs):
        user_rows = [[user_id, user_rec[0], user_rec[1]] for user_rec in user_recs]
        all_rows += user_rows
    result = pd.DataFrame(all_rows, columns=["query_id", "doc_id", "score"])
    result["query_id"] = result.query_id.astype('str')
    result["doc_id"] = result['doc_id'].astype('str')
    return result