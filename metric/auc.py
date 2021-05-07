
from sklearn.metrics import roc_auc_score

def auc(label_lis, pred_lis, pos_threshold=0.5):
    """

    :return: auc 的值

    # 1. ret 表示 「label = 1 的位置和」, m * (m + 1) / 2.0 表示 「label = 1 互相匹配的对数」 + 「rank 从 1开始计数」 
    # 2. 则 ret - m * (m + 1) / 2.0 代表正序对对数。如 [1, 0, 1, 0, 0, 0], 对于第一个1，正序对就有4个.
    # 3. m * (len(preds) - m) 表示 label = 1 和 label = 0 之间最多凑出的对数 (C(m1, 1) * C(m0, 1))
    # 4. 所以这是AUC的逆序对算法。
    """
    preds = []
    for _, label in zip(pred_lis, label_lis):
        preds.append( (_, label) )

    rank = len(preds)
    nums_pos, pos_rank_sum = 0, 0
    preds.sort(key=lambda x: x[0], reverse=True)

    for _, label in preds:
        if label > pos_threshold:
            nums_pos += 1
            pos_rank_sum += rank
        rank -= 1

    return ((pos_rank_sum - nums_pos * (nums_pos + 1) / 2.0) / (nums_pos * (len(preds) - nums_pos))) \
        if nums_pos > 0 and (len(preds) - nums_pos) > 0 else 0

if __name__=="__main__":
    label_lis, pred_lis = [1,1,0], [0.4, 0.6, 0.5]
    print("self auc: ", auc(label_lis, pred_lis))
    print("sklearn auc: ", roc_auc_score(label_lis, pred_lis))