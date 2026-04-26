#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CSHOGI_NATIVE
#include "init.hpp"
#include "position.hpp"
#include "cshogi.h"
#endif

namespace py = pybind11;

namespace {

struct CppNode {
    std::uint64_t key = 0;
    py::object py_node = py::none();
    long long move_count = 0;
    double value = 0.0;
    double sum_value = 0.0;
    std::vector<int> child_move;
    std::vector<double> child_move_count;
    std::vector<double> child_score_sum;
    std::vector<double> child_policy;
    bool dirty = false;
};

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

std::unordered_map<std::uint64_t, CppNode> &dl_cpp_tree() {
    static auto *obj = new std::unordered_map<std::uint64_t, CppNode>();
    return *obj;
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


std::vector<double> vector_from_numeric_obj(const py::object &obj) {
    std::vector<double> out;
    if (obj.is_none()) {
        return out;
    }

    if (py::isinstance<py::array>(obj)) {
        const py::array arr = py::cast<py::array>(obj);
        const py::buffer_info info = arr.request();
        if (info.ndim != 1) {
            throw std::runtime_error("expected 1D numeric array");
        }
        out.resize(static_cast<size_t>(info.shape[0]));
        for (ssize_t i = 0; i < info.shape[0]; ++i) {
            out[static_cast<size_t>(i)] = read_numeric_1d(arr, info, i);
        }
        return out;
    }

    const py::sequence seq = obj.cast<py::sequence>();
    const ssize_t n = py::len(seq);
    out.reserve(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        out.push_back(py::cast<double>(seq[i]));
    }
    return out;
}


std::vector<int> vector_from_int_obj(const py::object &obj) {
    std::vector<int> out;
    if (obj.is_none()) {
        return out;
    }

    const py::sequence seq = obj.cast<py::sequence>();
    const ssize_t n = py::len(seq);
    out.reserve(static_cast<size_t>(n));
    for (ssize_t i = 0; i < n; ++i) {
        out.push_back(py::cast<int>(seq[i]));
    }
    return out;
}


CppNode build_cpp_node(std::uint64_t key, py::object py_node) {
    CppNode node;
    node.key = key;
    node.py_node = std::move(py_node);
    node.move_count = py::cast<long long>(node.py_node.attr("move_count"));
    node.value = py::cast<double>(node.py_node.attr("value"));
    node.sum_value = py::cast<double>(node.py_node.attr("sum_value"));

    const py::object child_move_obj = node.py_node.attr("child_move");
    if (is_none_or_empty(child_move_obj)) {
        return node;
    }

    node.child_move = vector_from_int_obj(child_move_obj);
    node.child_move_count = vector_from_numeric_obj(node.py_node.attr("child_move_count"));
    node.child_score_sum = vector_from_numeric_obj(node.py_node.attr("child_score_sum"));
    node.child_policy = vector_from_numeric_obj(node.py_node.attr("child_policy"));

    const size_t n = node.child_move.size();
    if (node.child_move_count.size() != n) {
        node.child_move_count.resize(n, 0.0);
    }
    if (node.child_score_sum.size() != n) {
        node.child_score_sum.resize(n, 0.0);
    }
    if (node.child_policy.size() != n) {
        node.child_policy.resize(n, 0.0);
    }
    return node;
}


CppNode &ensure_cpp_node(std::uint64_t key, const py::object &board) {
    auto it = dl_cpp_tree().find(key);
    if (it != dl_cpp_tree().end()) {
        return it->second;
    }

    const py::int_ key_obj(key);
    py::object py_node = py::none();
    if (dl_data_tree().contains(key_obj)) {
        py_node = dl_data_tree()[key_obj];
    } else {
        const py::tuple next_dl = get_dl_node_func()(board).cast<py::tuple>();
        py_node = next_dl[1];
        if (py_node.is_none()) {
            const std::string sfen = py::cast<std::string>(board.attr("sfen")());
            throw py::key_error("Board is not in dl_data_tree: " + sfen);
        }
        dl_data_tree()[key_obj] = py_node;
    }

    auto inserted = dl_cpp_tree().emplace(key, build_cpp_node(key, py_node));
    return inserted.first->second;
}


void sync_cpp_to_python() {
    for (auto &entry : dl_cpp_tree()) {
        CppNode &node = entry.second;
        if (!node.dirty) {
            continue;
        }

        node.py_node.attr("move_count") = py::int_(node.move_count);
        node.py_node.attr("sum_value") = py::float_(node.sum_value);

        if (!node.child_move.empty()) {
            py::object child_move_count_obj = node.py_node.attr("child_move_count");
            py::object child_score_sum_obj = node.py_node.attr("child_score_sum");
            for (size_t i = 0; i < node.child_move.size(); ++i) {
                const py::int_ idx(static_cast<py::ssize_t>(i));
                child_move_count_obj[idx] = py::float_(node.child_move_count[i]);
                child_score_sum_obj[idx] = py::float_(node.child_score_sum[i]);
            }
        }

        node.dirty = false;
    }
}


int select_max_ucb_child_cpp_node(const CppNode &node) {
    const ssize_t n = static_cast<ssize_t>(node.child_move.size());
    if (n <= 0) {
        return 0;
    }

    const double mc = static_cast<double>(node.move_count);

    int best_idx = 0;
    double best_ucb = -std::numeric_limits<double>::infinity();

    for (ssize_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const double cmc = node.child_move_count[idx];
        const double css = node.child_score_sum[idx];
        const double cp = node.child_policy[idx];

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


int select_max_ucb_child_cpp(py::object node) {
    const py::object child_move_obj = node.attr("child_move");
    const ssize_t n = py::len(child_move_obj);
    if (n <= 0) {
        return 0;
    }

    CppNode temp = build_cpp_node(0, std::move(node));
    return select_max_ucb_child_cpp_node(temp);
}


class PathKeyGuard {
public:
    PathKeyGuard(std::unordered_set<std::uint64_t> &path_keys, std::uint64_t key) : path_keys_(path_keys), key_(key) {
        path_keys_.insert(key_);
    }

    ~PathKeyGuard() {
        path_keys_.erase(key_);
    }

private:
    std::unordered_set<std::uint64_t> &path_keys_;
    std::uint64_t key_;
};


class MoveGuard {
public:
    explicit MoveGuard(py::object board) : board_(std::move(board)), active_(true) {}

    ~MoveGuard() {
        if (!active_) {
            return;
        }
        try {
            board_.attr("pop")();
        } catch (...) {
            // Suppress all exceptions in destructor.
        }
    }

    void release() {
        active_ = false;
    }

private:
    py::object board_;
    bool active_;
};


double search_impl(py::object board, std::uint64_t node_key, std::unordered_set<std::uint64_t> &path_keys) {
    if (path_keys.find(node_key) != path_keys.end()) {
        return 0.5;
    }

    CppNode &node = ensure_cpp_node(node_key, board);
    PathKeyGuard guard(path_keys, node_key);

    node.move_count += 1;
    node.dirty = true;

    if (node.child_move.empty()) {
        const py::int_ key_obj(node_key);
        if (!visited_nodes().contains(key_obj)) {
            node.py_node.attr("board") = board.attr("copy")();
            visited_nodes().add(key_obj);
        }
        return node.value;
    }

    const int search_node = select_max_ucb_child_cpp_node(node);
    const size_t search_idx = static_cast<size_t>(search_node);
    node.child_move_count[search_idx] += 1.0;
    node.dirty = true;

    const int move_int = node.child_move[search_idx];
    board.attr("push")(move_int);
    MoveGuard move_guard(board);
    const std::uint64_t next_board_key = py::cast<std::uint64_t>(board.attr("zobrist_hash")());

    if (path_keys.find(next_board_key) != path_keys.end()) {
        return 0.5;
    }

    if (use_cshogi_is_draw()) {
        const int draw = py::cast<int>(board.attr("is_draw")());
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

    const py::int_ next_key_obj(next_board_key);
    if (!dl_data_tree().contains(next_key_obj)) {
        // Lazy-load from Python-side storage first.
        const py::tuple next_dl = get_dl_node_func()(board).cast<py::tuple>();
        py::object loaded_node = next_dl[1];
        if (!loaded_node.is_none()) {
            dl_data_tree()[next_key_obj] = loaded_node;
            dl_cpp_tree().emplace(next_board_key, build_cpp_node(next_board_key, loaded_node));
        } else {
            // If not found even after lazy lookup, create a provisional leaf node.
            py::object new_node = node_class_obj()();
            new_node.attr("board") = board.attr("copy")();
            new_node.attr("child_move") = py::none();
            new_node.attr("value") = py::float_(1.0 - node.value);
            dl_data_tree()[next_key_obj] = new_node;
            dl_cpp_tree().emplace(next_board_key, build_cpp_node(next_board_key, new_node));
        }
    }

    CppNode &next_node = ensure_cpp_node(next_board_key, board);
    if (next_node.child_move.empty()) {
        depth0_count() += 1;
    }

    double value = search_impl(board, next_board_key, path_keys);
    value = 1.0 - value;

    node.sum_value += value;
    node.child_score_sum[search_idx] += value;
    node.dirty = true;

    move_guard.release();
    board.attr("pop")();

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
    dl_cpp_tree().clear();

    for (auto item : dl_data_tree()) {
        const std::uint64_t key = py::cast<std::uint64_t>(item.first);
        py::object node = py::reinterpret_borrow<py::object>(item.second);
        dl_cpp_tree().emplace(key, build_cpp_node(key, node));
    }

    py::module cshogi = py::module::import("cshogi");
    not_repetition() = py::cast<int>(cshogi.attr("NOT_REPETITION"));
    repetition_draw() = py::cast<int>(cshogi.attr("REPETITION_DRAW"));
    repetition_win() = py::cast<int>(cshogi.attr("REPETITION_WIN"));
    repetition_superior() = py::cast<int>(cshogi.attr("REPETITION_SUPERIOR"));
}


double search_cpp(py::object node, py::object path_keys_obj = py::none()) {
    py::object board = node.attr("board");
    const std::uint64_t node_key = py::cast<std::uint64_t>(board.attr("zobrist_hash")());
    std::unordered_set<std::uint64_t> path_keys;

    if (!path_keys_obj.is_none()) {
        py::set py_path_keys = path_keys_obj.cast<py::set>();
        for (auto key : py_path_keys) {
            path_keys.insert(py::cast<std::uint64_t>(key));
        }
    }

    return search_impl(board, node_key, path_keys);
}


long long get_depth0_count() {
    return depth0_count();
}


#ifdef USE_CSHOGI_NATIVE
namespace {
    bool s_native_initialized = false;
    void ensure_native_init() {
        if (!s_native_initialized) {
            Position::initZobrist();
            s_native_initialized = true;
        }
    }

    static inline void rtrim(std::string &s) {
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
            s.pop_back();
        }
    }
}


py::tuple parse_book_cpp(const std::string &filepath, py::object node_class) {
    ensure_native_init();

    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open book file: " + filepath);
    }

    struct ChildEntry {
        std::string move_usi;
        int score;
        int depth;  // -1 means None
    };

    struct BookEntry {
        std::uint64_t key;
        std::string sfen;
        std::vector<ChildEntry> children;
    };

    std::vector<BookEntry> entries;
    entries.reserve(2500000);

    Position pos;
    std::string line;

    // Skip first line (header)
    std::getline(ifs, line);

    while (std::getline(ifs, line)) {
        rtrim(line);
        if (line.empty()) continue;

        if (line.compare(0, 4, "sfen") == 0) {
            entries.emplace_back();
            BookEntry &entry = entries.back();

            std::string sfen = line.substr(5);
            pos.set(sfen);
            entry.key = pos.getKey();
            entry.sfen = pos.toSFEN();
        } else if (!entries.empty()) {
            BookEntry &entry = entries.back();

            std::istringstream iss(line);
            ChildEntry child;
            std::string dummy;
            iss >> child.move_usi >> dummy >> child.score;

            child.depth = -1;
            std::string token;
            while (iss >> token) {
                try {
                    size_t pos_end = 0;
                    int val = std::stoi(token, &pos_end);
                    if (pos_end == token.size()) {
                        child.depth = val;
                        break;
                    }
                } catch (...) {
                    continue;
                }
            }

            entry.children.push_back(std::move(child));
        }
    }

    // Build Python dicts
    py::dict book_tree_out;
    py::dict book_child_depth_tree_out;

    for (auto &entry : entries) {
        py::int_ key_obj(entry.key);

        py::object node = node_class();
        node.attr("board") = py::cast(entry.sfen);

        const size_t n = entry.children.size();

        py::list child_move_list;
        for (auto &child : entry.children) {
            child_move_list.append(py::cast(child.move_usi));
        }
        node.attr("child_move") = child_move_list;

        // child_move_count: np.zeros(n)
        auto child_move_count = py::array_t<double>(static_cast<py::ssize_t>(n));
        std::memset(child_move_count.mutable_data(), 0, n * sizeof(double));
        node.attr("child_move_count") = child_move_count;

        // child_score: np.array(scores, dtype=float32)
        auto child_score = py::array_t<float>(static_cast<py::ssize_t>(n));
        {
            float *ptr = child_score.mutable_data();
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<float>(entry.children[i].score);
            }
        }
        node.attr("child_score") = child_score;

        // child_score_sum: np.zeros(n, dtype=float32)
        auto child_score_sum = py::array_t<float>(static_cast<py::ssize_t>(n));
        std::memset(child_score_sum.mutable_data(), 0, n * sizeof(float));
        node.attr("child_score_sum") = child_score_sum;

        book_tree_out[key_obj] = node;

        // book_child_depth_tree
        py::list depth_list;
        for (auto &child : entry.children) {
            if (child.depth == -1) {
                depth_list.append(py::none());
            } else {
                depth_list.append(py::cast(child.depth));
            }
        }
        book_child_depth_tree_out[key_obj] = depth_list;
    }

    return py::make_tuple(book_tree_out, book_child_depth_tree_out);
}

py::tuple parse_eval_sfens_cpp(const std::string &filepath) {
    ensure_native_init();

    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open book file: " + filepath);
    }

    std::vector<std::string> sfens;
    std::vector<std::uint64_t> keys;
    sfens.reserve(1000000);
    keys.reserve(1000000);

    std::string line;
    // Skip first line (header)
    std::getline(ifs, line);

    Position pos;
    Position rotated_pos;

    while (std::getline(ifs, line)) {
        rtrim(line);
        if (line.empty()) continue;

        if (line.compare(0, 4, "sfen") == 0) {
            std::string sfen = line.substr(5);

            pos.set(sfen);
            std::uint64_t key = pos.getKey();

            std::string rotated_sfen = __rotate_sfen(sfen);
            rotated_pos.set(rotated_sfen);
            std::uint64_t rotated_key = rotated_pos.getKey();

            std::uint64_t canonical_key = std::min(key, rotated_key);

            sfens.push_back(pos.toSFEN());
            keys.push_back(canonical_key);
        }
    }

    // Build Python list of sfens
    py::list sfen_list;
    for (auto &s : sfens) {
        sfen_list.append(py::cast(s));
    }

    // Build numpy array of canonical keys
    auto canonical_keys = py::array_t<std::uint64_t>(static_cast<py::ssize_t>(keys.size()));
    std::memcpy(canonical_keys.mutable_data(), keys.data(), keys.size() * sizeof(std::uint64_t));

    return py::make_tuple(sfen_list, canonical_keys);
}
#endif


PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS core (B1 full-search path)";
    m.def("init", &init, "Initialize module globals");
    m.def("search_cpp", &search_cpp, py::arg("node"), py::arg("path_keys") = py::none(), "Run recursive MCTS search");
    m.def("select_max_ucb_child_cpp", &select_max_ucb_child_cpp, "Select child with max UCB");
    m.def("sync_cpp_to_python", &sync_cpp_to_python, "Sync C++ node stats to Python nodes");
    m.def("get_depth0_count", &get_depth0_count, "Get depth0 count");
#ifdef USE_CSHOGI_NATIVE
    m.def("parse_book_cpp", &parse_book_cpp, py::arg("filepath"), py::arg("node_class"),
          "Parse book file using native C++ Position");
    m.def("parse_eval_sfens_cpp", &parse_eval_sfens_cpp, py::arg("filepath"),
          "Parse book file and return sfens with canonical keys");
#endif
}
