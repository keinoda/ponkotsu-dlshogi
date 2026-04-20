#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

double &c_puct() {
    static double value = 0.1;
    return value;
}

py::dict &dl_data_tree() {
    static auto *obj = new py::dict();
    return *obj;
}

py::dict &book_tree() {
    static auto *obj = new py::dict();
    return *obj;
}

py::set &visited_nodes() {
    static auto *obj = new py::set();
    return *obj;
}

bool &use_cshogi_is_draw() {
    static bool value = false;
    return value;
}

py::object &get_dl_node_func() {
    static auto *obj = new py::object(py::none());
    return *obj;
}

py::object &get_book_node_func() {
    static auto *obj = new py::object(py::none());
    return *obj;
}

py::object &node_class_obj() {
    static auto *obj = new py::object(py::none());
    return *obj;
}

int &not_repetition() {
    static int value = 0;
    return value;
}

int &repetition_draw() {
    static int value = 1;
    return value;
}

int &repetition_win() {
    static int value = 2;
    return value;
}

int &repetition_superior() {
    static int value = 3;
    return value;
}

long long &depth0_count() {
    static long long value = 0;
    return value;
}


double score_to_value_cpp(double score, double a = 756.0864962951762) {
    return 1.0 / (1.0 + std::exp(-score / a));
}


bool is_none_or_empty(const py::object &obj) {
    if (obj.is_none()) {
        return true;
    }

    try {
        return py::len(obj) == 0;
    } catch (const py::error_already_set &) {
        return false;
    }
}


double read_numeric_1d(const py::array &arr, const py::buffer_info &info, ssize_t index) {
    const char *ptr = static_cast<const char *>(info.ptr) + index * info.strides[0];

    if (info.format == py::format_descriptor<float>::format()) {
        return static_cast<double>(*reinterpret_cast<const float *>(ptr));
    }
    if (info.format == py::format_descriptor<double>::format()) {
        return *reinterpret_cast<const double *>(ptr);
    }
    if (info.format == py::format_descriptor<std::int32_t>::format()) {
        return static_cast<double>(*reinterpret_cast<const std::int32_t *>(ptr));
    }
    if (info.format == py::format_descriptor<std::int64_t>::format()) {
        return static_cast<double>(*reinterpret_cast<const std::int64_t *>(ptr));
    }

    // Fallback for unknown dtypes.
    return py::cast<double>(arr[py::int_(index)]);
}


int select_max_ucb_child_cpp(py::object node) {
    const py::object child_move_obj = node.attr("child_move");
    const ssize_t n = py::len(child_move_obj);
    if (n <= 0) {
        return 0;
    }

    const py::array cmc_arr = py::cast<py::array>(node.attr("child_move_count"));
    const py::array css_arr = py::cast<py::array>(node.attr("child_score_sum"));
    const py::array cp_arr = py::cast<py::array>(node.attr("child_policy"));

    const py::buffer_info cmc_info = cmc_arr.request();
    const py::buffer_info css_info = css_arr.request();
    const py::buffer_info cp_info = cp_arr.request();

    if (cmc_info.ndim != 1 || css_info.ndim != 1 || cp_info.ndim != 1) {
        throw std::runtime_error("child arrays must be 1D");
    }

    const double mc = py::cast<double>(node.attr("move_count"));

    int best_idx = 0;
    double best_ucb = -std::numeric_limits<double>::infinity();

    for (ssize_t i = 0; i < n; ++i) {
        const double cmc = read_numeric_1d(cmc_arr, cmc_info, i);
        const double css = read_numeric_1d(css_arr, css_info, i);
        const double cp = read_numeric_1d(cp_arr, cp_info, i);

        const double q = (cmc != 0.0) ? (css / cmc) : 0.0;
        const double u = (mc == 0.0) ? 1.0 : std::sqrt(mc / (1.0 + cmc));
        const double ucb = q + c_puct() * cp * u;

        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_idx = static_cast<int>(i);
        }
    }

    return best_idx;
}


class PathKeyGuard {
public:
    PathKeyGuard(py::set &path_keys, py::object key) : path_keys_(path_keys), key_(std::move(key)) {
        path_keys_.add(key_);
    }

    ~PathKeyGuard() {
        try {
            path_keys_.attr("discard")(key_);
        } catch (...) {
            // Suppress all exceptions in destructor.
        }
    }

private:
    py::set &path_keys_;
    py::object key_;
};


double search_impl(py::object node, py::set &path_keys) {
    const py::object node_key = node.attr("board").attr("zobrist_hash")();
    if (path_keys.contains(node_key)) {
        return 0.5;
    }

    PathKeyGuard guard(path_keys, node_key);

    node.attr("move_count") = py::int_(py::cast<long long>(node.attr("move_count")) + 1);

    const py::object child_move_obj = node.attr("child_move");
    if (is_none_or_empty(child_move_obj)) {
        visited_nodes().attr("add")(node_key);
        return py::cast<double>(node.attr("value"));
    }

    const int search_node = select_max_ucb_child_cpp(node);
    const py::int_ search_idx(search_node);

    py::object child_move_count_obj = node.attr("child_move_count");
    child_move_count_obj[search_idx] = py::float_(py::cast<double>(child_move_count_obj[search_idx]) + 1.0);

    py::object next_board = node.attr("board").attr("copy")();
    const int move_int = py::cast<int>(child_move_obj[search_idx]);
    next_board.attr("push")(move_int);
    const py::object next_board_key = next_board.attr("zobrist_hash")();

    if (path_keys.contains(next_board_key)) {
        return 0.5;
    }

    if (use_cshogi_is_draw()) {
        const int draw = py::cast<int>(next_board.attr("is_draw")());
        if (draw != not_repetition()) {
            if (draw == repetition_draw()) {
                return 0.5;
            }
            if (draw == repetition_win() || draw == repetition_superior()) {
                return 1.0;
            }
            return 0.0;
        }
    }

    if (!dl_data_tree().contains(next_board_key)) {
        const py::tuple current_dl = get_dl_node_func()(node.attr("board")).cast<py::tuple>();
        const py::object current_dl_node = current_dl[1];
        if (current_dl_node.is_none()) {
            const std::string sfen = py::cast<std::string>(node.attr("board").attr("sfen")());
            throw py::key_error("Current board is not in dl_data_tree: " + sfen);
        }

        py::object new_node = node_class_obj()();
        new_node.attr("board") = next_board.attr("copy")();
        new_node.attr("child_move") = py::none();
        new_node.attr("value") = py::float_(1.0 - py::cast<double>(current_dl_node.attr("value")));
        dl_data_tree()[next_board_key] = new_node;
    }

    py::object next_node_from_tree = dl_data_tree()[next_board_key];
    if (is_none_or_empty(next_node_from_tree.attr("child_move"))) {
        const py::tuple current_book = get_book_node_func()(node.attr("board")).cast<py::tuple>();
        const py::object current_book_node = current_book[1];

        if (!current_book_node.is_none()) {
            const py::object current_book_child_moves = current_book_node.attr("child_move");
            const py::object selected_move = child_move_obj[search_idx];

            const bool contains = py::cast<bool>(current_book_child_moves.attr("__contains__")(selected_move));
            if (contains) {
                const int idx = py::cast<int>(current_book_child_moves.attr("index")(selected_move));
                const double score = py::cast<double>(current_book_node.attr("child_score")[py::int_(idx)]);
                next_node_from_tree.attr("value") = py::float_(1.0 - score_to_value_cpp(score));
            }
        }
        depth0_count() += 1;
    }

    const py::tuple next_dl = get_dl_node_func()(next_board).cast<py::tuple>();
    const py::object next_node = next_dl[1];
    if (next_node.is_none()) {
        const std::string sfen = py::cast<std::string>(next_board.attr("sfen")());
        throw py::key_error("Next board is not in dl_data_tree: " + sfen);
    }

    next_node.attr("board") = next_board;

    double value = search_impl(next_node, path_keys);
    value = 1.0 - value;

    node.attr("sum_value") = py::float_(py::cast<double>(node.attr("sum_value")) + value);

    py::object child_score_sum_obj = node.attr("child_score_sum");
    child_score_sum_obj[search_idx] = py::float_(py::cast<double>(child_score_sum_obj[search_idx]) + value);

    return value;
}

}  // namespace


void init(py::dict dl_tree_obj,
        py::dict book_tree_obj,
        py::set visited_nodes_obj,
        bool use_cshogi_is_draw_flag,
        py::object get_dl_func,
        py::object get_book_func,
        py::object node_class) {
    dl_data_tree() = std::move(dl_tree_obj);
    book_tree() = std::move(book_tree_obj);
    visited_nodes() = std::move(visited_nodes_obj);
    use_cshogi_is_draw() = use_cshogi_is_draw_flag;
    get_dl_node_func() = std::move(get_dl_func);
    get_book_node_func() = std::move(get_book_func);
    node_class_obj() = std::move(node_class);
    depth0_count() = 0;

    py::module cshogi = py::module::import("cshogi");
    not_repetition() = py::cast<int>(cshogi.attr("NOT_REPETITION"));
    repetition_draw() = py::cast<int>(cshogi.attr("REPETITION_DRAW"));
    repetition_win() = py::cast<int>(cshogi.attr("REPETITION_WIN"));
    repetition_superior() = py::cast<int>(cshogi.attr("REPETITION_SUPERIOR"));
}


double search_cpp(py::object node, py::object path_keys_obj = py::none()) {
    py::set path_keys = path_keys_obj.is_none() ? py::set() : path_keys_obj.cast<py::set>();
    return search_impl(std::move(node), path_keys);
}


long long get_depth0_count() {
    return depth0_count();
}


PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS core (B1 full-search path)";
    m.def("init", &init, "Initialize module globals");
    m.def("search_cpp", &search_cpp, py::arg("node"), py::arg("path_keys") = py::none(), "Run recursive MCTS search");
    m.def("select_max_ucb_child_cpp", &select_max_ucb_child_cpp, "Select child with max UCB");
    m.def("get_depth0_count", &get_depth0_count, "Get depth0 count");
}
