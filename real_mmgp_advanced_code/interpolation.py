import bisect

def BinarySearch(orderedList, item):
    return max(bisect.bisect_right(orderedList, item) - 1, 0)


def PieceWiseLinearInterpolation(item, itemIndices, vectors, tolerance = 1e-4):
    if item <= itemIndices[0]:
        return vectors[0]
    if item >= itemIndices[-1]:
        return vectors[-1]


    prev = BinarySearch(itemIndices, item)
    coef = (item - itemIndices[prev]) / (itemIndices[prev+1] - itemIndices[prev])

    if 0.5 - abs(coef - 0.5) < tolerance:
        coef = round(coef)

    return (
            coef * vectors[prev+1]
            + (1 - coef) * vectors[prev]
        )

def PieceWiseLinearInterpolationVectorized(items, itemIndices, vectors):
    return [PieceWiseLinearInterpolation(item, itemIndices, vectors) for item in items]


