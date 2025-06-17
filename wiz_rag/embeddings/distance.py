from numpy import dot
from numpy.linalg import norm


# 余弦相似度计算
def cosine_similarity(a, b):
    """
    余弦相似度计算：
    向量点积 / 向量模长相乘
    """
    return dot(a, b) / (norm(a) * norm(b))


# 欧几里得距离
def euclidean_distance(a, b):
    """
    欧几里得距离计算
    差异平方和
    """
    return norm(a - b)
