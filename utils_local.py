import numpy as np

def get_order(graph):
    """Orientation of tangent bundle Q."""
    N = graph.shape[0]
    nnz = graph.nnz
    indptr = graph.indptr
    indices = graph.indices
    data = graph.data
    indptr_enlarged = np.empty(N+1, dtype=np.int32)
    indices_enlarged = np.empty(nnz+N, dtype=np.int32)
    data_enlarged = np.empty(nnz+N)
    for i in range(N):
        delta = indptr[i+1] - indptr[i]
        indices_local = indices[indptr[i]:indptr[i+1]]
        data_local = data[indptr[i]:indptr[i+1]]
        order_local = np.argsort(data_local)

        shift = indptr[i] + i
        indices_enlarged[shift:shift+delta] = indices_local[order_local]
        indices_enlarged[shift+delta] = -1
        data_enlarged[shift:shift+delta] = data_local[order_local]
        data_enlarged[shift+delta] = np.inf
        indptr_enlarged[i] = shift
    indptr_enlarged[N] = indptr[N] + N

    points_oriented = np.empty(N, dtype=np.int32)
    points_oriented[0] = 0
    points_order = np.empty(N, dtype=np.int32)
    points_order[0] = 0
    points_close = np.empty(N, dtype=np.int32)
    points_close[0] = -1
    points_used = np.zeros(N, dtype=np.bool_)
    points_used[0] = True
    for i in range(1, N):
        points_local = points_oriented[:i]
        min_distances = np.empty_like(points_local, dtype=np.float64)
        for p, point in enumerate(points_local):
            min_distances[p] = data_enlarged[indptr_enlarged[point]]
        point_min = np.argmin(min_distances)
        point_close = points_local[point_min]
        point_selected = indices_enlarged[indptr_enlarged[point_close]]
        
        while points_used[point_selected] and point_selected >= 0:
            indptr_enlarged[point_close] += 1
            min_distances[point_min] = data_enlarged[indptr_enlarged[point_close]]
            point_min = np.argmin(min_distances)
            point_close = points_local[point_min]
            point_selected = indices_enlarged[indptr_enlarged[point_close]]

        if point_selected >= 0:
            indptr_enlarged[point_close] += 1
        elif point_selected == -1:
            print("Another component is found")
            point_close = 0
            for p, point in enumerate(points_local):
                if p != point:
                    point_selected = p
                    break
            if point_selected == -1:
                point_selected = i
        else:
            print("Error in function get_order")

        k = np.searchsorted(points_local, point_selected)
        points_oriented[k+1:i+1] = points_oriented[k:i]
        points_oriented[k] = point_selected
        points_order[i] = point_selected
        points_close[i] = point_close
        points_used[point_selected] = True
    
    return points_order, points_close
