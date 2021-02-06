import numpy as np
from scipy.spatial.distance import mahalanobis
import math
import re
from collections import Counter


WORD = re.compile(r'\w+')


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_vectors(text1, text2):
    return text_to_vector(text1), text_to_vector(text2)


# Serve per la minkowski_distance
def nth_root(value, n_root):

    root_value = 1/float(n_root)
    return value**root_value


def jaccard_similarity(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection_cardinality = len(set.intersection(
        *[set(vec1.keys()), set(vec2.keys())]))
    union_cardinality = len(set.union(*[set(vec1.keys()), set(vec2.keys())]))
    return intersection_cardinality/float(union_cardinality)


def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# Generalizzazione delle varie distanze: euclidea, manhattan etc.
def minkowski_distance(text1, text2, power):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    distance = nth_root(sum(abs((vec1[x] - vec2[x]))**power for x in intersection) + sum(vec1[x]**power for x in set(vec1.keys()).difference(intersection)) +
                        sum(vec2[x]**power for x in set(vec2.keys()).difference(intersection)), power)

    return distance


def __get_covariance_matrix(X, Y):
    covariance = np.cov([X, Y], rowvar=0)
    return covariance


def __get_pseudo_inverse(M):
    if M.ndim > 1:
        inverse_covariance = np.linalg.pinv(M)
    else:
        eps = 1e-3
        inverse_covariance = 1/(eps + M)
    return inverse_covariance


def mahalanobis_distance(text1, text2):
    vec1, vec2 = get_vectors(text1, text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    intersection_vec1 = np.array([vec1[x] for x in intersection])
    intersection_vec2 = np.array([vec2[x] for x in intersection])
    covariance_matrix = __get_covariance_matrix(
        intersection_vec1, intersection_vec2)
    inverse_covariance = __get_pseudo_inverse(covariance_matrix)
    return mahalanobis(intersection_vec1, intersection_vec2, inverse_covariance)
