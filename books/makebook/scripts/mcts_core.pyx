# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport sqrt, exp

from cshogi import NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR

cdef float _c_puct = 0.1

# Module-level references (set via init())
dl_data_tree = None
book_tree = None
visited_nodes = None
depth0_count = 0
use_cshogi_is_draw = False
_get_dl_node = None
_get_book_node = None
_Node = None


def init(dl_tree, book, visited, cshogi_draw, get_dl_func, get_book_func, NodeClass):
    global dl_data_tree, book_tree, visited_nodes, use_cshogi_is_draw
    global _get_dl_node, _get_book_node, _Node, depth0_count
    dl_data_tree = dl_tree
    book_tree = book
    visited_nodes = visited
    use_cshogi_is_draw = cshogi_draw
    _get_dl_node = get_dl_func
    _get_book_node = get_book_func
    _Node = NodeClass
    depth0_count = 0


def get_depth0_count():
    return depth0_count


cpdef float score_to_value_cy(float score, float a=756.0864962951762):
    return 1.0 / (1.0 + exp(-score / a))


cpdef int select_max_ucb_child_cy(object node):
    cdef int n = len(node.child_move)
    cdef int i, best_idx = 0
    cdef float q, u_val, ucb, best_ucb = -1e30
    cdef int mc = node.move_count
    cdef float cmc, css, cp

    for i in range(n):
        cmc = node.child_move_count[i]
        css = node.child_score_sum[i]
        cp = node.child_policy[i]

        q = css / cmc if cmc != 0.0 else 0.0
        u_val = 1.0 if mc == 0 else sqrt(<float>mc / (1.0 + cmc))
        ucb = q + _c_puct * cp * u_val

        if ucb > best_ucb:
            best_ucb = ucb
            best_idx = i

    return best_idx


cpdef float search_cy(object node, object path_keys=None):
    global depth0_count

    if path_keys is None:
        path_keys = set()

    node_key = node.board.zobrist_hash()
    if node_key in path_keys:
        return 0.5

    path_keys.add(node_key)
    node.move_count += 1

    cdef float value
    cdef int search_idx
    cdef int move_int

    try:
        if not node.child_move:
            visited_nodes.add(node_key)
            return <float>node.value

        search_idx = select_max_ucb_child_cy(node)
        node.child_move_count[search_idx] += 1

        next_board = node.board.copy()
        move_int = <int>node.child_move[search_idx]
        next_board.push(move_int)
        next_board_key = next_board.zobrist_hash()

        if next_board_key in path_keys:
            return 0.5

        if use_cshogi_is_draw:
            draw = next_board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    return 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    return 1.0
                else:
                    return 0.0

        if next_board_key not in dl_data_tree:
            _, current_dl_node = _get_dl_node(node.board)
            if current_dl_node is None:
                raise KeyError(f"Current board is not in dl_data_tree: {node.board.sfen()}")

            new_node = _Node()
            new_node.board = next_board.copy()
            new_node.child_move = None
            new_node.value = 1.0 - <float>current_dl_node.value
            dl_data_tree[next_board_key] = new_node

        if not dl_data_tree[next_board_key].child_move:
            _, current_book_node = _get_book_node(node.board)
            if current_book_node is not None:
                child_move_val = node.child_move[search_idx]
                if child_move_val in current_book_node.child_move:
                    idx = current_book_node.child_move.index(child_move_val)
                    dl_data_tree[next_board_key].value = 1.0 - score_to_value_cy(
                        current_book_node.child_score[idx]
                    )
            depth0_count += 1

        _, next_node = _get_dl_node(next_board)
        if next_node is None:
            raise KeyError(f"Next board is not in dl_data_tree: {next_board.sfen()}")

        next_node.board = next_board
        value = search_cy(next_node, path_keys)
        value = 1.0 - value

        node.sum_value += value
        node.child_score_sum[search_idx] += value
        return value
    finally:
        path_keys.remove(node_key)
