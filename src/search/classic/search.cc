/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "search/classic/search.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits> // Required for numeric_limits
#include <optional> // Added for std::optional
#include <sstream>
#include <thread>
#include <vector> // Added for std::vector

#include "chess/chess.h"     // Added for Value, GameResult etc.
#include "chess/position.h" // Added for PositionHash
#include "neural/encoder.h"
#include "search/classic/node.h"
#include "search/search_stopper.h" // Added for StoppersHints
#include "search/stats.h"          // Added for IterationStats
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"

// NOTE: The compilation error regarding "utils/bitops.h" suggests a missing file
// or an issue with the build system's include paths. This cannot be fixed
// within this file itself. Ensure the file exists and is correctly located
// relative to where it's included (e.g., in trainingdata.h).

namespace lczero {
namespace classic {

// Moved constants inside the namespace
constexpr Value kValueKnownWin = kValueMate;   // Use Value directly
constexpr Value kValueKnownLoss = -kValueMate; // Use Value directly

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          false, &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  float GetMUtility(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_ || !child) return 0.0f; // Added null check
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      // This allows a smooth M effect with higher q thresholds, which is
      // necessary for using MLH together with contempt.
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    // Use GetValue() which corresponds to WL, consistent with Min/Max logic base cases
    return parent && std::abs(parent->GetValue()) > q_threshold; // Added null check
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

}  // namespace

Search::Search(const NodeTree& tree, Backend* backend,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      backend_(backend),
      backend_attributes_(backend->GetAttributes()),
      params_(options), // Pass options to SearchParams constructor
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_ ? root_node_->GetN() : 0), // Handle null root_node_
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
    // Ensure root_node_ is not null before accessing members
    if (!root_node_) {
        // Handle error: cannot initialize search without a root node
        throw Exception("Search initialized with null root node");
    }
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      // For infinite search disable contempt, only "white"/"black" make sense.
      contempt_mode_ = ContemptMode::NONE;
      // Issue a warning only if contempt mode would have an effect.
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        if (uci_responder_) uci_responder_->OutputThinkingInfo(&info); // Added null check
      }
    } else {
      // Otherwise set it to the root move's side, unless pondering.
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }

  // NOTE: Load known_win/loss flags when constructing node from TT
  // This logic would typically be in Node::Node or a TT probing function.
  // Example placeholder:
  /*
  if (tt_entry) {
      root_node_->is_known_win.store(tt_entry->known_win, std::memory_order_relaxed);
      root_node_->is_known_loss.store(tt_entry->known_loss, std::memory_order_relaxed);
      // ... load other TT data ...
  }
  */
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  if (!node || !node->HasChildren()) return; // Add null/children check
  float total = 0;
  std::vector<float> noise;
  noise.reserve(node->GetNumEdges()); // Reserve memory

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  // Avoid division by zero or near-zero
  if (total < std::numeric_limits<float>::epsilon()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    if (edge && noise_idx < static_cast<int>(noise.size())) { // Add null/bounds check
        edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
    } else {
        // Handle error or break if mismatch
        break;
    }
  }
}
}  // namespace

namespace {
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert,
                         float max_reasonable_s) {
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    // Avoid division by zero for ratio
    if (std::abs(wdl_rescale_ratio) < 1e-9f) {
        wdl_rescale_ratio = 1e-9f * FastSign(wdl_rescale_ratio); // Use small non-zero value
    }
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    // Avoid division by zero for s
    if (std::abs(a+b) < 1e-9f) return 0; // Or handle error appropriately
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Originally hardcoded, made into a parameter for piece odds.
    if (!invert) s = std::min(max_reasonable_s, s);
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      s = std::min(max_reasonable_s, s);
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    // Avoid division by zero for logistic
    if (std::abs(s_new) < 1e-9f) return 0; // Or handle appropriately
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  if (!root_node_) return; // Add null check

  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ > 0 ? total_playouts_ : 1); // Avoid division by zero
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  // Calculate default values safely, handling potential null root_node_ (though checked above)
  const auto default_wl = root_node_ ? -root_node_->GetWL() : 0.0f;
  const auto default_d = root_node_ ? root_node_->GetD() : 1.0f; // Default draw if root is null/unvisited
  const auto default_q = - (root_node_ ? root_node_->GetQ(-draw_score) : 0.0f);

  for (const auto& edge : edges) {
    if (!edge) continue; // Skip null edges if any

    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      mu_uci = WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true, params_.GetWDLMaxS());
    }
    const auto q = edge.GetQ(default_q, draw_score);

    // Prioritize reporting proven mates if available
    bool is_known_win = edge.node() && edge.node()->is_known_win.load(std::memory_order_relaxed);
    bool is_known_loss = edge.node() && edge.node()->is_known_loss.load(std::memory_order_relaxed);

    // Mate score calculation needs default M value if node is null or N=0
    float default_m = root_node_ ? root_node_->GetM() : 0.0f;
    float edge_m = edge.GetM(default_m); // Get M safely

    if (is_known_loss) { // Child is known loss -> I win mate
        // Ensure M is somewhat reasonable before calculating mate score
        uci_info.mate = static_cast<int>(std::round(edge_m)/2 + 1);
    } else if (is_known_win) { // Child is known win -> I lose mate
        uci_info.mate = -static_cast<int>(std::round(edge_m)/2 + 1);
    } else if (edge.IsTerminal() && std::abs(wl) > 1e-6f) { // Original terminal logic (check wl != 0 with tolerance)
      uci_info.mate = std::copysign(
          std::round(edge_m) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
       // Avoid division by zero/large numbers
       float denom = (1 - 0.976953126 * std::pow(wl, 14));
       uci_info.score = (std::abs(denom) > 1e-9f) ? (295 * wl / denom) : (wl > 0 ? 30000 : -30000); // Assign large value if denom is near zero
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    } else if (score_type == "WDL_mu") {
      // Reports the WDL mu value whenever it is reasonable, and defaults to
      // centipawn otherwise.
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 45 * tan(1.56728071628 * wl);
      uci_info.score =
          backend_attributes_.has_wdl && std::abs(mu_uci) > 1e-6f && // Check mu_uci != 0
                  std::abs(wl) + d < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }

    auto wdl_w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) {
      wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2));
      wdl_l = 1000 - wdl_w;
      wdl_d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (backend_attributes_.has_mlh) {
      // Calculate moves_left safely
      float root_m = root_node_ ? root_node_->GetM() : 0.0f;
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_m)) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  if (uci_responder_) uci_responder_->OutputThinkingInfo(&uci_infos); // Added null check
}


void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ > 0 ? total_playouts_ : 1)) || // Avoid division by zero
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
       if (uci_responder_) uci_responder_->OutputThinkingInfo(&info); // Added null check
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}

namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  if (!node) return params.GetFpuValue(is_root_node); // Return default FPU if node is null

  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             // Use GetValue() here, which is WL, consistent with Min/Max logic base cases
             : -node->GetValue() -
                   value * std::sqrt(node->GetVisitedPolicy());
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
   if (!node) return params.GetFpuValue(is_root_node); // Return default FPU if node is null

  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetValue() - value * std::sqrt(visited_pol);
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  // Avoid log of zero or negative
  if (N + base <= 0) return init;
  return init + (k != 0.0f ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  if (!node) return {}; // Add null check

  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<EdgeAndNode> edges;
  if (node->HasChildren()){ // Check if children exist before iterating
      for (const auto& edge : node->Edges()) edges.push_back(edge);
  }


  std::sort(edges.begin(), edges.end(),
            [this, &fpu, &U_coeff, &draw_score](EdgeAndNode a, EdgeAndNode b) { // Added `this` capture for params_
              float score_a = -std::numeric_limits<float>::infinity();
              float score_b = -std::numeric_limits<float>::infinity();
              bool overridden_a = false;
              bool overridden_b = false;

              if (params_.GetProvenStateHandling()) {
                  if(a.node() && a.node()->is_known_loss.load(std::memory_order_relaxed)) { score_a = kValueKnownWin + static_cast<float>(a.GetN()); overridden_a = true; }
                  else if (a.node() && a.node()->is_known_win.load(std::memory_order_relaxed)) { score_a = kValueKnownLoss - static_cast<float>(a.GetN()); overridden_a = true; }

                  if(b.node() && b.node()->is_known_loss.load(std::memory_order_relaxed)) { score_b = kValueKnownWin + static_cast<float>(b.GetN()); overridden_b = true; }
                  else if (b.node() && b.node()->is_known_win.load(std::memory_order_relaxed)) { score_b = kValueKnownLoss - static_cast<float>(b.GetN()); overridden_b = true; }
              }

              if (!overridden_a) score_a = a.GetQ(fpu, draw_score) + a.GetU(U_coeff);
              if (!overridden_b) score_b = b.GetQ(fpu, draw_score) + b.GetU(U_coeff);

              // Sort primarily by PUCT score (descending), then N (descending)
              // Add tolerance for float comparison
              if (std::abs(score_a - score_b) > 1e-6f) return score_a > score_b;
              return a.GetN() > b.GetN();
            });

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const Node* n) { // Use const Node*
    const auto sign = n == node ? 1 : -1; // Sign depends on perspective RELATIVE TO PARENT NODE
                                          // If n==node (self), sign is 1 (current player perspective)
                                          // If n is child, sign is -1 (opponent perspective relative to parent)
    if (n) {
      auto wl = sign * n->GetWL(); // Apply sign for perspective shift
      auto d = n->GetD(); // D doesn't change perspective
      auto m = n->GetM();
      // Use GetQ with the CORRECT draw score perspective for the node `n`
      // If n==node, depth is node's depth. If n is child, depth is node's depth + 1.
      float node_draw_score = GetDrawScore(is_odd_depth != (n == node)); // Flip parity if child
      auto q = n->GetQ(node_draw_score); // Get node's Q with its correct draw score
      auto is_perspective = ((contempt_mode_ == ContemptMode::BLACK) ==
                             played_history_.IsBlackToMove())
                                ? 1.0f
                                : -1.0f;
     // Display potentially rescaled values if relevant
     float display_wl = wl; // Use perspective-adjusted wl
     float display_d = d;
      if (params_.GetWDLRescaleDiff() != 0.0f && contempt_mode_ != ContemptMode::NONE){
         // WDLRescale expects value from WHITE's perspective? Or player to move? Check definition.
         // Assuming it expects value from player-to-move perspective (already in wl)
         WDLRescale(
            display_wl, display_d, params_.GetWDLRescaleRatio(),
            params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
            is_perspective * sign, // Apply perspective adjustments
            true, params_.GetWDLMaxS());
      }

      print(oss, "(WL: ", display_wl, ") ", 8, 5);
      print(oss, "(D: ", display_d, ") ", 5, 3);
      print(oss, "(M: ", m, ") ", 4, 1);
      print(oss, "(Q: ", q, ") ", 8, 5); // Display node's internal Q
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
      print(oss, "(Q: ", fpu, ") ", 8, 5); // Display FPU for unvisited
    }
  };
  auto print_tail = [&](auto* oss, const Node* n) { // Use const Node*
     const auto sign = n == node ? 1 : -1; // Sign for perspective shift
     std::optional<Value> v; // Use Value (double)

     if (n && n->is_known_win.load(std::memory_order_relaxed)) {
          v = sign * kValueKnownWin; // Apply sign
     } else if (n && n->is_known_loss.load(std::memory_order_relaxed)) {
          v = sign * kValueKnownLoss; // Apply sign
     } else if (n && n->IsTerminal()) {
         v = sign * n->GetWL(); // Use WL for terminal state value display, apply sign
     } else if (n) { // Check only if n exists
         std::optional<EvalResult> nneval = GetCachedNNEval(n);
         if (nneval) {
             // NN eval is from node's perspective, apply sign for parent view
             v = sign * nneval->q;
         }
     }

     if (v) {
         print(oss, "(V: ", *v, ") ", 7, 4);
     } else {
         *oss << "(V:  -.----) ";
     }


    if (n) {
        // Display proven state flags
        if(n->is_known_win.load(std::memory_order_relaxed)) *oss << "(KW)";
        else if(n->is_known_loss.load(std::memory_order_relaxed)) *oss << "(KL)";
        // Display original bounds logic
        else {
            auto [lo, up] = n->GetBounds();
            // Bounds are from node's perspective, flip for parent's view if needed
            if (sign == -1) { // If printing child from parent perspective
                 lo = FlipGameResult(lo); up = FlipGameResult(up); std::swap(lo, up);
            }
             *oss << (lo == up                                                ? "(T) " // Proven Terminal (Win/Loss/Draw)
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) " // Can't Lose (can draw or win)
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) " // Can't Win (can draw or lose)
                                                                       : "");   // Regular/Unknown
        }
    }
  };


  std::vector<std::string> infos;
  const auto m_evaluator =
      backend_attributes_.has_mlh ? MEvaluator(params_, node) : MEvaluator();
  for (const auto& edge : edges) {
    if (!edge) continue; // Skip null edges

    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    float U = edge.GetU(U_coeff);
    float score_override = -std::numeric_limits<float>::infinity();
    bool overridden = false;

    // Recalculate score override for display consistency
    if (params_.GetProvenStateHandling()) {
        Node* child_node = edge.node();
        if (child_node && child_node->is_known_loss.load(std::memory_order_relaxed)) {
            score_override = kValueKnownWin + static_cast<float>(edge.GetN());
            overridden = true;
        } else if (child_node && child_node->is_known_win.load(std::memory_order_relaxed)) {
            score_override = kValueKnownLoss - static_cast<float>(edge.GetN());
            overridden = true;
        }
    }

    std::ostringstream oss;
    oss << std::left;
    // Ensure move is valid before converting
    Move move = edge.GetMove(is_black_to_move);
    if (move == Move::Null()) continue;

    print_head(&oss, move.ToString(true),
               MoveToNNIndex(move, played_history_.Last().GetBoard().IsBlackToMove()), // Use correct side to move
               edge.GetN(),
               edge.GetNInFlight(), edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", U, ") ", 6, 5);
    print(&oss, "(S: ", overridden ? score_override : (Q + U + M), ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss_node;
  print_head(&oss_node, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss_node, node); // Pass node itself
  print_tail(&oss_node, node); // Pass node itself
  infos.emplace_back(oss_node.str());
  return infos;
}


void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  if (!root_node_) return; // Add null check

  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
     if (uci_responder_) uci_responder_->OutputThinkingInfo(&infos); // Added null check
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  if (root_node_->HasChildren()){ // Check if children exist
      for (auto& edge : root_node_->Edges()) {
        if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
          continue;
        }
        if (edge.HasNode()) {
          LOGFILE << "--- Opponent moves after: " << final_bestmove_.ToString(true);
          for (const auto& line : GetVerboseStats(edge.node())) {
            LOGFILE << line;
          }
        }
      }
  }
}


PositionHistory Search::GetPositionHistoryAtNode(const Node* node) const {
  PositionHistory history(played_history_);
  if (!node || node == root_node_) return history; // Handle null or root node

  std::vector<Move> rmoves;
  for (const Node* n = node; n != root_node_ && n != nullptr; n = n->GetParent()) { // Added null check for n
    Edge* own_edge = n->GetOwnEdge();
    if (!own_edge) break; // Should not happen if n!=root_node_
    rmoves.push_back(own_edge->GetMove());
  }
  for (auto it = rmoves.rbegin(); it != rmoves.rend(); it++) {
    history.Append(*it);
  }
  return history;
}

namespace {
std::vector<Move> GetNodeLegalMoves(const Node* node, const ChessBoard& board) {
  if (!node) return board.GenerateLegalMoves(); // Generate if node is null

  std::vector<Move> moves;
  if (node && node->HasChildren()) {
    moves.reserve(node->GetNumEdges());
    // Iterate safely
    for(const auto& edge : node->Edges()) {
         if(edge) moves.push_back(edge.GetMove());
    }
    // Verify generated moves match edges if node exists? Optional.
    return moves;
  }
  // If node exists but has no children (leaf node), generate moves
  return board.GenerateLegalMoves();
}
}  // namespace


std::optional<EvalResult> Search::GetCachedNNEval(const Node* node) const {
  if (!node) return {};
  PositionHistory history = GetPositionHistoryAtNode(node);
  std::vector<Move> legal_moves =
      GetNodeLegalMoves(node, history.Last().GetBoard());
  return backend_->GetCachedEvaluation(
      EvalPosition{history.GetPositions(), legal_moves});
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded or null.
  if (!root_node_ || total_playouts_ + initial_visits_ == 0) return; // Added null check

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    if (uci_responder_) uci_responder_->OutputBestMove(&info); // Added null check
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_.Reset(); // Use Reset() instead of assigning empty EdgeAndNode
  }
}


Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!root_node_) return {0.0f, 1.0f, 0.0f}; // Return default if root is null

  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);

  if (!best_edge) { // Handle case where no best child found
      if (move) *move = Move();
      if (is_terminal) *is_terminal = root_node_->IsTerminal(); // Use root's terminal status
      return {parent_wl, parent_d, parent_m}; // Return parent's eval
  }

  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());

  bool known_win = best_edge.node() && best_edge.node()->is_known_win.load(std::memory_order_relaxed);
  bool known_loss = best_edge.node() && best_edge.node()->is_known_loss.load(std::memory_order_relaxed);

  if (is_terminal) *is_terminal = best_edge.IsTerminal() || known_win || known_loss;

  // Return known win/loss value if available
  if (known_loss){ // Child known loss -> I win
      // Calculate M safely
      float child_m = best_edge.GetM(parent_m -1);
      return {kValueKnownWin, 0.0f, child_m + 1};
  }
  if (known_win){ // Child known win -> I lose
      float child_m = best_edge.GetM(parent_m -1);
      return {kValueKnownLoss, 0.0f, child_m + 1};
  }

  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (!root_node_ || root_node_->GetN() == 0) return; // Add null check
  if (!root_node_->HasChildren()) return; // Check children exist

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature > 1e-6f && decay_moves > 0) { // Check temperature > 0 and decay_moves > 0
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  EdgeAndNode bestmove_edge;
  if (temperature > 1e-6f) { // Use tolerance for float comparison
       bestmove_edge = GetBestRootChildWithTemperature(temperature);
  } else {
       bestmove_edge = GetBestChildNoTemperature(root_node_, 0);
  }

  // Handle case where no best move was found
  if (!bestmove_edge) {
      LOGFILE << "Warning: EnsureBestMoveKnown could not find a best edge.";
      // Fallback: pick the absolute best without temperature if possible
      bestmove_edge = GetBestChildNoTemperature(root_node_, 0);
      if (!bestmove_edge) { // Still no edge? Major issue or no legal moves.
          final_bestmove_ = Move();
          final_pondermove_ = Move();
          return;
      }
  }

  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  if (bestmove_edge.GetN() > 0 && bestmove_edge.node() && bestmove_edge.node()->HasChildren()) { // Added null check for node
    EdgeAndNode ponder_edge = GetBestChildNoTemperature(bestmove_edge.node(), 1);
    final_pondermove_ = ponder_edge ? ponder_edge.GetMove(!played_history_.IsBlackToMove()) : Move(); // Handle null ponder edge
  } else {
      final_pondermove_ = Move(); // Set null ponder move
  }
}


std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  if (!parent || parent->GetN() == 0) return {}; // Add null check
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);

  std::vector<EdgeAndNode> edges;
  if (parent->HasChildren()){ // Check if children exist
      for (auto& edge : parent->Edges()) {
        if (!edge) continue; // Skip null edges
        if (parent == root_node_ && !root_move_filter_.empty() &&
            std::find(root_move_filter_.begin(), root_move_filter_.end(),
                      edge.GetMove()) == root_move_filter_.end()) {
          continue;
        }
        edges.push_back(edge);
      }
  }

  if (edges.empty()) return {}; // Return empty if no valid edges found

  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [this, draw_score](const auto& a, const auto& b) { // Added this capture for params_
        // The function returns "true" when a is preferred to b.

        // --- NEW: Prioritize proven states ---
        bool a_known_loss = a.node() && a.node()->is_known_loss.load(std::memory_order_relaxed);
        bool a_known_win = a.node() && a.node()->is_known_win.load(std::memory_order_relaxed);
        bool b_known_loss = b.node() && b.node()->is_known_loss.load(std::memory_order_relaxed);
        bool b_known_win = b.node() && b.node()->is_known_win.load(std::memory_order_relaxed);

        // Prefer known wins (opponent loss)
        if (a_known_loss && !b_known_loss) return true;
        if (!a_known_loss && b_known_loss) return false;
        // Avoid known losses (opponent win)
        if (a_known_win && !b_known_win) return false;
        if (!a_known_win && b_known_win) return true;

        // If both are known loss (win for me), prefer shorter M (faster mate)
        if (a_known_loss && b_known_loss) {
             // Handle cases where M might be default (0) if N=0
            float m_a = a.GetM(0.0f);
            float m_b = b.GetM(0.0f);
            return m_a < m_b;
        }
        // If both are known win (loss for me), prefer longer M (delay mate)
        if (a_known_win && b_known_win) {
             float m_a = a.GetM(0.0f);
             float m_b = b.GetM(0.0f);
             return m_a > m_b;
        }
        // --- End of new proven state logic ---

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // Use GetWL safely with default
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || std::abs(wl) < 1e-6f) { // Use tolerance for wl==0
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal (or both same terminal type), use standard rule.
        // Prefer largest playouts then eval then prior.
        if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
        // Use GetQ for comparison as it includes draw score
        // Use tolerance for float comparison
        float q_a = a.GetQ(0.0f, draw_score);
        float q_b = b.GetQ(0.0f, draw_score);
        if (std::abs(q_a - q_b) > 1e-6f) {
            return q_a > q_b;
        }
        // Use tolerance for policy comparison
        float p_a = a.GetP();
        float p_b = b.GetP();
        if (std::abs(p_a - p_b) > 1e-9f) { // Smaller tolerance for policy
             return p_a > p_b;
        }
        // If everything is equal, potentially use move order or other tie-break
        return false; // Indicate equality if all else fails


        // This part is now unreachable due to the comprehensive checks above
        /*
        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }
        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
        */
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}


EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}


EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  if (!root_node_) return {}; // Add null check

  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);

  std::vector<float> cumulative_sums;
  std::vector<int> eligible_indices; // Store indices of eligible edges
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -std::numeric_limits<float>::infinity(); // Use -infinity
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  if (!root_node_->HasChildren()) return {}; // Return if no children

  // First pass: find max visits and max eval among eligible moves
  int edge_index = -1;
  for (auto& edge : root_node_->Edges()) {
    edge_index++;
    if (!edge) continue; // Skip null iterators

    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    // --- NEW: Skip proven losing moves ---
    if (params_.GetProvenStateHandling() && edge.node() && edge.node()->is_known_win.load(std::memory_order_relaxed)) {
        continue; // Avoid selecting moves leading to known loss
    }

    float current_n = static_cast<float>(edge.GetN()) + offset;
    if (current_n > max_n) {
      max_n = current_n;
    }
    max_eval = std::max(max_eval, edge.GetQ(fpu, draw_score));
    eligible_indices.push_back(edge_index); // Store index if potentially eligible
  }

  // Handle case where max_eval remains -infinity (no valid moves found initially)
  if (eligible_indices.empty()) {
       LOGFILE << "Warning: No eligible moves found in first pass for temperature selection.";
       return GetBestChildNoTemperature(root_node_, 0); // Fallback
  }


  // Second pass: calculate probabilities for moves within the cutoff
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  edge_index = -1; // Reset index
  std::vector<int> final_eligible_indices; // Indices passing the eval cutoff

  if (root_node_->HasChildren()){ // Check again before iterating
      for (auto& edge : root_node_->Edges()) {
          edge_index++;
          if (!edge) continue; // Skip null iterators

           // Quick check if index was potentially eligible from first pass
           bool potentially_eligible = false;
           for(int idx : eligible_indices) { if(idx == edge_index) { potentially_eligible = true; break; } }
           if (!potentially_eligible) continue;

          // Re-check filters and proven state
          if (!root_move_filter_.empty() &&
              std::find(root_move_filter_.begin(), root_move_filter_.end(),
                        edge.GetMove()) == root_move_filter_.end()) {
            continue;
          }
          if (params_.GetProvenStateHandling() && edge.node() && edge.node()->is_known_win.load(std::memory_order_relaxed)) {
              continue;
          }

          if (edge.GetQ(fpu, draw_score) < min_eval) continue;

          // Calculate probability contribution
          float prob_component = std::pow(
              std::max(0.0f,
                       (max_n <= 1e-6f // Use tolerance for max_n near zero
                            ? edge.GetP() // Use policy prior if no visits
                            // Use normalized visit count relative to max_n
                            : ((static_cast<float>(edge.GetN()) + offset) / max_n))),
              1.0f / temperature); // Avoid division by zero temperature

          if (prob_component > 1e-9f) { // Only include if contribution is significant
              sum += prob_component;
              cumulative_sums.push_back(sum);
              final_eligible_indices.push_back(edge_index); // Store index passing cutoff
          }
      }
  }


  // Handle case where no moves pass the eval cutoff
  if (cumulative_sums.empty()) {
       LOGFILE << "Warning: No eligible moves passed eval cutoff for temperature selection, falling back to best move.";
       return GetBestChildNoTemperature(root_node_, 0); // Fallback
  }
  assert(sum > 1e-9f); // Should have positive sum if not empty

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int chosen_cumulative_idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

   // Ensure index is within bounds
   if (chosen_cumulative_idx < 0 || chosen_cumulative_idx >= static_cast<int>(final_eligible_indices.size())) {
        LOGFILE << "Error: Invalid index chosen during temperature selection.";
        return GetBestChildNoTemperature(root_node_, 0); // Fallback
   }

   int chosen_edge_index = final_eligible_indices[chosen_cumulative_idx];

   // Third pass: find the edge corresponding to the chosen index
   edge_index = -1;
   if (root_node_->HasChildren()){ // Check again
       for (auto& edge : root_node_->Edges()) {
           edge_index++;
           if (!edge) continue;
           if (edge_index == chosen_edge_index) return edge;
       }
   }

  assert(false); // Should have found the move
  return GetBestChildNoTemperature(root_node_, 0); // Fallback
}


void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  if (how_many == 0 && threads_.empty()) {
    how_many = backend_attributes_.suggested_num_search_threads +
               !backend_attributes_.runs_on_cpu;
  }
  thread_count_.store(how_many, std::memory_order_release);
  // First thread is a watchdog thread.
  if (threads_.empty()) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this]() {
      SearchWorker worker(this, params_);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ > 0 ? total_playouts_ : 1); // Avoid division by zero
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit or is null, none of this code is safe.
  if (root_node_ && root_node_->GetN() > 0 && root_node_->HasChildren()) { // Add checks
    const float draw_score = GetDrawScore(false); // Use root's draw score perspective
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -std::numeric_limits<float>::infinity(); // Use -infinity
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = backend_attributes_.has_mlh
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      if (!edge) continue; // Skip null edges

      const auto n = edge.GetN();
      // Use GetQ with root's draw score for consistency
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);

      // Check for proven states
      bool is_known_win = edge.node() && edge.node()->is_known_win.load(std::memory_order_relaxed);
      bool is_known_loss = edge.node() && edge.node()->is_known_loss.load(std::memory_order_relaxed);

      if (is_known_loss) { // Child known loss means I win
          stats->win_found = true;
          // Calculate mate depth safely
          float child_m = edge.GetM(0.0f);
           stats->mate_depth = std::min(stats->mate_depth,
                                static_cast<int>(std::round(child_m)) / 2 + 1);
      } else if (is_known_win) { // Child known win means I lose
          stats->num_losing_edges += 1;
      } else if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) { // Original terminal logic
        stats->win_found = true;
        if (std::abs(edge.GetWL(0.0f) - 1.0f) < 1e-6f && !edge.IsTbTerminal()) { // Mate found via game rules
             float child_m = edge.GetM(0.0f);
            stats->mate_depth =
                std::min(stats->mate_depth,
                         static_cast<int>(std::round(child_m)) / 2 + 1);
        }
      } else if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }

      // Resign logic adjusted for known states
      if (n > 0 && q > -0.98f && !is_known_loss) { // Don't resign if a known win exists
        stats->may_resign = false;
      }

      // Max N / Max Q+M logic remains the same
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      // Use tolerance for float comparison
      if (q_plus_m > max_q_plus_m + 1e-6f) {
           max_q_plus_m = q_plus_m;
           max_n_has_max_q_plus_m = (max_n == n);
      } else if (std::abs(q_plus_m - max_q_plus_m) < 1e-6f && max_n == n) {
          max_n_has_max_q_plus_m = true; // If Q+M is equal and N is max, it holds
      }

    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}


void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // Handle case where remaining_time might be zero or negative
    remaining_time = std::max((int64_t)kMinWaitTimeMs, remaining_time);

    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
     if (threads_.back().joinable()) { // Check before joining
        threads_.back().join();
     }
    threads_.pop_back();
  }
}


void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    if (!node) continue; // Add null check
    for (node = node->GetParent(); node != nullptr && node != root_node_->GetParent(); // Add null check for node
         node = node->GetParent()) {
      node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();
  }
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

SearchWorker::SearchWorker(Search* search, const SearchParams& params)
    : search_(search),
      params_(params),
      computation_(search_->backend_->RequestComputation(params.GetBatchMerging())),
      moves_left_support_(search_->backend_attributes_.has_mlh),
      task_workers_(params_.GetMaxConcurrentPickers()),
      history_(search->played_history_) {} // Initialize history

SearchWorker::~SearchWorker() {
    if (!task_workspace_.vtp_buffer.empty()) {
        LOGFILE << "Warning: vtp_buffer not empty in SearchWorker destructor.";
    }
    if (!task_workspace_.visits_to_perform.empty()) {
         LOGFILE << "Warning: visits_to_perform not empty in SearchWorker destructor.";
    }
    // Clean up any remaining picking tasks if necessary
    {
        Mutex::Lock lock(picking_tasks_mutex_);
        picking_tasks_.clear();
        task_count_.store(0, std::memory_order_release);
    }
    if (picker_thread_.joinable()) {
        picker_thread_.join();
    }
}


void SearchWorker::RunBlocking() {
  if (task_workers_ > 0) {
    picker_thread_ = std::thread([this]() { RunTasks(1); });
  }
  RunTasks(0);
  if (picker_thread_.joinable()) picker_thread_.join();
}


void SearchWorker::RunTasks(int tid) {
  const int collision_limit = params_.GetMaxCollisionEvents();
  while (!search_->stop_.load(std::memory_order_acquire)) {
    if (tid == 0) {
      int pending = 0;
      if (search_->params_.GetMaxConcurrentSearchers() != 0) {
        pending = search_->pending_searchers_.load(std::memory_order_acquire);
        if (pending == 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
      }
      ExecuteOneIteration();
      if (search_->params_.GetMaxConcurrentSearchers() != 0) {
        search_->pending_searchers_.fetch_sub(1, std::memory_order_acq_rel);
      }
    } else {
      int count = WaitForTasks();
      if (count == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        ProcessPickedTask(0, count, &task_workspace_);
      }
    }
  }
}


void SearchWorker::ExecuteOneIteration() {
  InitializeIteration(std::move(computation_));
  GatherMinibatch();
  MaybePrefetchIntoCache();
  RunNNComputation();
  FetchMinibatchResults();
  DoBackupUpdate();
  UpdateCounters();
  computation_ = search_->backend_->RequestComputation(params_.GetBatchMerging());
}


void SearchWorker::InitializeIteration(std::unique_ptr<BackendComputation> computation) {
    computation_ = std::move(computation);
    minibatch_.clear();
    number_out_of_order_ = 0;
}


void SearchWorker::GatherMinibatch() {
  if (search_->stop_.load(std::memory_order_acquire)) return;
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    search_->CancelSharedCollisions();
  }
  if (task_workers_ > 0) {
    ProcessPickedTask(0, WaitForTasks(), &task_workspace_);
  }
  PickNodesToExtend(params_.GetMaxCollisionEvents());
  CollectCollisions();
}

void SearchWorker::ProcessPickedTask(int start_idx, int end_idx, TaskWorkspace* workspace) {
  std::vector<PickingTask> tasks;
  {
    Mutex::Lock lock(picking_tasks_mutex_);
    for (int i = start_idx; i < end_idx; ++i) {
        if (i < static_cast<int>(picking_tasks_.size())) { // Bounds check
            tasks.emplace_back(std::move(picking_tasks_[i]));
        }
    }
    if (!tasks.empty()) {
      picking_tasks_.erase(picking_tasks_.begin() + start_idx,
                           picking_tasks_.begin() + end_idx);
    }
  }
  for (auto& task : tasks) {
    PickNodesToExtendTask(task.node, task.base_depth, task.visits, task.moves,
                           &minibatch_, workspace);
  }
}


void SearchWorker::ResetTasks() {
    Mutex::Lock lock(picking_tasks_mutex_);
    picking_tasks_.clear();
    task_count_.store(0, std::memory_order_release);
    task_added_.notify_all(); // Notify any waiting threads
}


int SearchWorker::WaitForTasks() {
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.wait(lock.get_raw(), [this]() {
        return task_count_.load(std::memory_order_acquire) > 0 ||
               search_->stop_.load(std::memory_order_acquire);
    });
    int count = task_count_.load(std::memory_order_acquire);
    if (count > 0) {
        task_count_.store(0, std::memory_order_release);
    }
    return count;
}


void SearchWorker::PickNodesToExtend(int collision_limit) {
  ResetTasks();
  history_.Trim(search_->played_history_.GetLength());
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    PickNodesToExtendTask(search_->root_node_, 0, collision_limit, {},
                          &minibatch_, &task_workspace_);
  }
}

// Ensure node flags are correctly set if its parent is terminal/known.
void SearchWorker::EnsureNodeTwoFoldCorrectForDepth(Node* child_node, int depth) {
    if (!child_node) return;
    Node* parent = child_node->GetParent();
    if (!parent) return; // Should not happen if child_node is valid

    bool parent_known_win = parent->is_known_win.load(std::memory_order_relaxed);
    bool parent_known_loss = parent->is_known_loss.load(std::memory_order_relaxed);
    bool parent_terminal = parent->IsTerminal();

    // If parent is known win, child is known loss
    if (parent_known_win && !child_node->is_known_loss.load(std::memory_order_relaxed)) {
        child_node->is_known_loss.store(true, std::memory_order_relaxed);
        // Optionally make terminal if not already? Depends on game logic.
        // if (!child_node->IsTerminal()) child_node->MakeTerminal(GameResult::BLACK_WON, parent->GetM() + 1);
    }
    // If parent is known loss, child is known win
    else if (parent_known_loss && !child_node->is_known_win.load(std::memory_order_relaxed)) {
         child_node->is_known_win.store(true, std::memory_order_relaxed);
        // if (!child_node->IsTerminal()) child_node->MakeTerminal(GameResult::WHITE_WON, parent->GetM() + 1);
    }
    // If parent is terminal draw, child is terminal draw
    else if (parent_terminal && parent->GetValue() == kValueDraw && !child_node->IsTerminal()) {
        // Child leads back to a drawn position (3-fold like)
        child_node->MakeTerminal(GameResult::DRAW, parent->GetM() + 1);
    }

    // Add standard 2-fold repetition check
    if (!child_node->IsTerminal() && !child_node->is_known_win.load(std::memory_order_relaxed) && !child_node->is_known_loss.load(std::memory_order_relaxed)) {
        // Need history up to the child node to check repetitions
        PositionHistory temp_history = search_->GetPositionHistoryAtNode(child_node);
        if (temp_history.Last().GetRepetitions() >= 1) { // Check for 2-fold (>=1 repetition)
            // Usually 3-fold is draw, 2-fold might not be terminal unless specific rules apply
            // Standard MCTS doesn't make node terminal on 2-fold, but could adjust value (e.g., towards draw)
            // Here, we'll just log it for now, or implement specific logic if needed.
             // LOGFILE << "Node " << child_node->GetHash() << " reached via 2-fold repetition.";
        }
    }
}


void SearchWorker::PickNodesToExtendTask(
    Node* node, int base_depth, int collision_limit, // Corrected parameter order
    const std::vector<Move>& moves_to_base,
    std::vector<NodeToProcess>* receiver,
    TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS {

    // --- Access proven state handling option ---
    const bool proven_state_enabled = params_.GetProvenStateHandling();

    auto& vtp_buffer = workspace->vtp_buffer;
    auto& visits_to_perform = workspace->visits_to_perform;
    visits_to_perform.clear();
    auto& vtp_last_filled = workspace->vtp_last_filled;
    vtp_last_filled.clear();
    auto& current_path = workspace->current_path;
    current_path.clear();
    auto& moves_to_path = workspace->moves_to_path;
    moves_to_path = moves_to_base;

    if (receiver->capacity() < 30) {
        receiver->reserve(receiver->size() + 30);
    }

    std::array<float, 256> current_pol;
    std::array<float, 256> current_util;
    std::array<float, 256> current_score;
    std::array<int, 256> current_nstarted;
    auto& cur_iters = workspace->cur_iters;

    Node::Iterator best_edge;
    Node::Iterator second_best_edge;
    // Use current_best_edge_ safely
    int64_t best_node_n = 0;
    { // Lock only for reading current_best_edge_
        SharedMutex::SharedLock lock(search_->nodes_mutex_);
        best_node_n = search_->current_best_edge_.GetN();
    }


    int passed_off = 0;
    int completed_visits = 0;

    bool is_root_node = node == search_->root_node_;
    const float even_draw_score = search_->GetDrawScore(false);
    const float odd_draw_score = search_->GetDrawScore(true);
    const auto& root_move_filter = search_->root_move_filter_;
    auto m_evaluator = moves_left_support_ ? MEvaluator(params_, node) : MEvaluator(); // Pass node to MEvaluator

    int max_limit = std::numeric_limits<int>::max();

    current_path.push_back(-1);
    while (current_path.size() > 0) {
        if (current_path.back() == -1) {
            int cur_limit = collision_limit - completed_visits - passed_off;
            if (current_path.size() > 1) {
                if (current_path.size() - 2 < visits_to_perform.size() && current_path[current_path.size() - 2] >= 0 && current_path[current_path.size() - 2] < 256){ // Added bounds check >=0
                    cur_limit = (*visits_to_perform[current_path.size() - 2])[current_path[current_path.size() - 2]];
                } else {
                    cur_limit = collision_limit - completed_visits - passed_off;
                }
            }
            if (cur_limit <= 0) {
                node = node ? node->GetParent() : nullptr; // Null check
                if (!moves_to_path.empty()) moves_to_path.pop_back();
                current_path.pop_back();
                if (!visits_to_perform.empty()) {
                    vtp_buffer.push_back(std::move(visits_to_perform.back()));
                    visits_to_perform.pop_back();
                    vtp_last_filled.pop_back();
                }
                continue;
            }

            // --- Check Proven State ---
            if (proven_state_enabled && node &&
                (node->is_known_win.load(std::memory_order_relaxed) || node->is_known_loss.load(std::memory_order_relaxed))) {
                 if (cur_limit > 0) {
                   receiver->push_back(NodeToProcess::Collision(
                       node, static_cast<uint16_t>(current_path.size() + base_depth),
                       cur_limit, 0));
                   completed_visits += cur_limit;
                 }
                 node = node->GetParent();
                 if (!moves_to_path.empty()) moves_to_path.pop_back();
                 current_path.pop_back();
                 if (!visits_to_perform.empty()) {
                     vtp_buffer.push_back(std::move(visits_to_perform.back()));
                     visits_to_perform.pop_back();
                     vtp_last_filled.pop_back();
                 }
                 continue;
            }
            // --- End Proven State Check ---

            if (!node || node->GetN() == 0 || node->IsTerminal()) { // Add null check
                 if (is_root_node && node) {
                     if (node->TryStartScoreUpdate()) {
                         int visit_count = std::min(1, cur_limit);
                         if (visit_count > 0) {
                             receiver->emplace_back(NodeToProcess::Visit(
                                 node, static_cast<uint16_t>(current_path.size() + base_depth)));
                             completed_visits += visit_count;
                             cur_limit -= visit_count;
                         } else {
                              node->CancelScoreUpdate(1);
                         }
                     }
                 }
                 if (cur_limit > 0) {
                     int max_count = 0;
                     if (base_depth == 0 && max_limit > cur_limit) {
                         max_count = max_limit;
                     }
                     if (node) {
                         receiver->push_back(NodeToProcess::Collision(
                             node, static_cast<uint16_t>(current_path.size() + base_depth),
                             cur_limit, max_count));
                         completed_visits += cur_limit;
                     }
                 }
                 node = node ? node->GetParent() : nullptr;
                 if (!moves_to_path.empty()) moves_to_path.pop_back();
                 current_path.pop_back();
                 if (!visits_to_perform.empty()) {
                     vtp_buffer.push_back(std::move(visits_to_perform.back()));
                     visits_to_perform.pop_back();
                     vtp_last_filled.pop_back();
                 }
                 continue;
            }

            if (is_root_node && node) {
                node->IncrementNInFlight(cur_limit);
            }

            if (vtp_buffer.size() > 0) {
                visits_to_perform.push_back(std::move(vtp_buffer.back()));
                vtp_buffer.pop_back();
            } else {
                visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
            }
            std::fill(visits_to_perform.back()->begin(), visits_to_perform.back()->end(), 0);
            vtp_last_filled.push_back(-1);


            int max_needed = node->GetNumEdges();
            if (!is_root_node || root_move_filter.empty()) {
                max_needed = std::min(max_needed, static_cast<int>(node->GetNStarted() + cur_limit + 2));
            }
            if (max_needed > 0 && node->HasChildren()) { // Check HasChildren
                 node->CopyPolicy(max_needed, current_pol.data());
                 for (int i = 0; i < max_needed; i++) {
                     current_util[i] = std::numeric_limits<float>::lowest();
                 }
            }

            const float draw_score = ((current_path.size() + base_depth) % 2 == 0)
                                         ? odd_draw_score
                                         : even_draw_score;
            m_evaluator.SetParent(node);
            float visited_pol = 0.0f;
            if (node->HasChildren()){
                 for (Node* child : node->VisitedNodes()) {
                     if (!child) continue;
                     int index = child->Index();
                     if(index >=0 && index < max_needed) {
                         visited_pol += current_pol[index];
                         float q = child->GetQ(draw_score); // Child Q needs opponent's draw score
                         current_util[index] = q + m_evaluator.GetMUtility(child, q);
                     }
                 }
            }
            const float fpu =
                GetFpu(params_, node, is_root_node, draw_score, visited_pol);
            for (int i = 0; i < max_needed; i++) {
                if (current_util[i] == std::numeric_limits<float>::lowest()) {
                      current_util[i] = fpu + m_evaluator.GetDefaultMUtility();
                }
            }

            const float cpuct = ComputeCpuct(params_, node->GetN(), is_root_node);
            const float puct_mult =
                cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
            int cache_filled_idx = -1;
            while (cur_limit > 0) {
                float best = std::numeric_limits<float>::lowest();
                int best_idx = -1;
                float best_without_u = std::numeric_limits<float>::lowest();
                float second_best = std::numeric_limits<float>::lowest();
                bool can_exit = false;
                best_edge.Reset();
                second_best_edge.Reset();

                if (max_needed > 0 && node->HasChildren()) {
                     for (int idx = 0; idx < max_needed; ++idx) {
                         if (idx > cache_filled_idx) {
                             if (idx == 0) {
                                 cur_iters[idx] = node->Edges();
                             } else {
                                  if (cur_iters[idx - 1]) {
                                     cur_iters[idx] = cur_iters[idx - 1];
                                     ++cur_iters[idx];
                                  } else {
                                     cur_iters[idx].Reset();
                                  }
                             }
                              if (cur_iters[idx]) {
                                 current_nstarted[idx] = cur_iters[idx].GetNStarted();
                              } else {
                                 current_nstarted[idx] = 0;
                                 if (idx <= cache_filled_idx) cache_filled_idx = idx-1;
                                 break;
                              }
                         }

                          if (!cur_iters[idx]) continue;

                         int nstarted = current_nstarted[idx];
                         const float util = current_util[idx];
                         if (idx > cache_filled_idx) {
                             current_score[idx] =
                                 current_pol[idx] * puct_mult / (1 + nstarted) + util;
                             cache_filled_idx++;
                         }

                         // --- Proven State Handling in Selection ---
                         float score = -std::numeric_limits<float>::infinity();
                         bool overridden = false;
                          if (proven_state_enabled) {
                             Node* child_node = cur_iters[idx].node();
                             if (child_node) {
                                 if (child_node->is_known_loss.load(std::memory_order_relaxed)) {
                                     score = kValueKnownWin + static_cast<float>(cur_iters[idx].GetN());
                                     overridden = true;
                                 } else if (child_node->is_known_win.load(std::memory_order_relaxed)) {
                                     score = kValueKnownLoss - static_cast<float>(cur_iters[idx].GetN());
                                     overridden = true;
                                 }
                             }
                          }

                         if (!overridden) {
                             score = current_score[idx];
                         }
                         // --- End Proven State Handling ---

                         if (is_root_node) {
                              // Pruning logic adjusted for potentially changing best_node_n
                             int64_t current_best_n = 0;
                             { // Lock only for reading current_best_edge_ N
                                 SharedMutex::SharedLock lock(search_->nodes_mutex_);
                                 current_best_n = search_->current_best_edge_.GetN();
                             }
                             if (cur_iters[idx] != search_->current_best_edge_ &&
                                 latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                                     current_best_n - cur_iters[idx].GetN()) {
                               continue;
                             }

                             if (!root_move_filter_.empty() &&
                                 std::find(root_move_filter_.begin(), root_move_filter_.end(),
                                           cur_iters[idx].GetMove()) == root_move_filter_.end()) {
                               continue;
                             }
                         }

                         if (score > best) {
                             second_best = best;
                             second_best_edge = best_edge;
                             best = score;
                             best_idx = idx;
                             best_without_u = overridden ? kValueKnownLoss : util;
                             best_edge = cur_iters[idx];
                         } else if (score > second_best) {
                             second_best = score;
                             second_best_edge = cur_iters[idx];
                         }

                         if (can_exit) break;
                         if (nstarted == 0) {
                             can_exit = true;
                         }
                     } // End for loop edges
                } // End if has children

                if (best_idx == -1) {
                     LOGFILE << "Warning: No best edge found during selection at depth " << current_path.size() + base_depth;
                     break;
                }

                 int new_visits = 0;
                 if (second_best_edge) {
                   int estimated_visits_to_change_best = std::numeric_limits<int>::max();
                   bool best_is_proven_win = proven_state_enabled && best > kValueKnownLoss + 100.f; // Increased threshold
                   bool best_is_proven_loss = proven_state_enabled && best < kValueKnownWin - 100.f; // Increased threshold

                   // Estimate visits only if best is not proven and needs changing
                   if (!best_is_proven_win && !best_is_proven_loss && best_without_u < second_best) {
                        const auto n1 = current_nstarted[best_idx] + 1;
                        // Ensure safety and avoid division by zero/small policy
                        if (second_best > best_without_u + 1e-9f && current_pol[best_idx] > 1e-9f) {
                              float denom = second_best - best_without_u;
                              if (denom > 1e-9f) { // Avoid division by near zero
                                   estimated_visits_to_change_best = static_cast<int>(
                                        std::max(1.0f, std::min(current_pol[best_idx] * puct_mult / denom - n1 + 1,
                                                            1e9f)));
                              }
                        }
                   }
                   max_limit = std::min(max_limit, estimated_visits_to_change_best);
                   new_visits = std::min(cur_limit, estimated_visits_to_change_best);
                 } else {
                   new_visits = cur_limit;
                 }
                 new_visits = std::max(0, new_visits); // Ensure non-negative

                if (visits_to_perform.empty() || vtp_last_filled.empty()) {
                     assert(false && "visits_to_perform or vtp_last_filled became empty unexpectedly");
                     break;
                }

                 if (best_idx >= 0 && best_idx < 256) {
                     // Only fill gaps if necessary
                     if (best_idx > vtp_last_filled.back()) {
                          auto* vtp_array = visits_to_perform.back()->data();
                          // Check bounds before filling
                          int fill_start = vtp_last_filled.back() + 1;
                          int fill_end = best_idx + 1;
                          if (fill_start >= 0 && fill_end <= 256 && fill_start < fill_end) {
                              std::fill(vtp_array + fill_start, vtp_array + fill_end, 0);
                          }
                          vtp_last_filled.back() = best_idx;
                     }
                     if (new_visits > 0) {
                        (*visits_to_perform.back())[best_idx] += new_visits;
                         // Update last filled index if this index is now the furthest
                         if (best_idx > vtp_last_filled.back()) {
                            vtp_last_filled.back() = best_idx;
                         }
                     }
                 } else {
                     assert(false && "Invalid best_idx encountered");
                     break;
                 }

                cur_limit -= new_visits;
                Node* child_node = best_edge.GetOrSpawnNode(node); // Pass parent node

                EnsureNodeTwoFoldCorrectForDepth(
                    child_node, current_path.size() + base_depth);

                bool decremented = false;
                if (child_node && child_node->TryStartScoreUpdate()) {
                     if (best_idx >= 0 && best_idx < 256) {
                           current_nstarted[best_idx]++;
                           // Only decrement new_visits if it was positive
                           if (new_visits > 0) new_visits -= 1;
                           decremented = true;
                     }
                     if (child_node->GetN() > 0 && !child_node->IsTerminal() && new_visits > 0) {
                          child_node->IncrementNInFlight(new_visits);
                           if (best_idx >= 0 && best_idx < 256) {
                               current_nstarted[best_idx] += new_visits;
                           }
                     }
                      if (best_idx >= 0 && best_idx < 256) {
                           current_score[best_idx] = current_pol[best_idx] * puct_mult /
                                                        (1 + current_nstarted[best_idx]) +
                                                    current_util[best_idx];
                      }
                }

                if (child_node && decremented &&
                    (child_node->GetN() == 0 || child_node->IsTerminal() || (proven_state_enabled && (child_node->is_known_win.load(std::memory_order_relaxed) || child_node->is_known_loss.load(std::memory_order_relaxed))) )) {
                     if (best_idx >= 0 && best_idx < 256) {
                         // Ensure value doesn't go below zero
                         (*visits_to_perform.back())[best_idx] = std::max(0, (*visits_to_perform.back())[best_idx] - 1);
                     }
                   receiver->emplace_back(NodeToProcess::Visit(
                       child_node,
                       static_cast<uint16_t>(current_path.size() + base_depth)));
                   completed_visits++;
                   receiver->back().moves_to_visit = moves_to_path;
                   if(best_edge) {
                      receiver->back().moves_to_visit.push_back(best_edge.GetMove());
                   }
                }
                // No need to update vtp_last_filled again here

            } // End while cur_limit > 0

            is_root_node = false;
             // Split logic
             if (!visits_to_perform.empty() && !vtp_last_filled.empty() && vtp_last_filled.back() >= 0 && node && node->HasChildren()) {
                 for (int i = 0; i <= vtp_last_filled.back(); i++) {
                      if (i < 256) {
                          int child_limit = (*visits_to_perform.back())[i];
                          if (task_workers_ > 0 && child_limit > 0 &&
                              child_limit > params_.GetMinimumWorkSizeForPicking() &&
                              child_limit < ((collision_limit - passed_off - completed_visits > 0) ? ((collision_limit - passed_off - completed_visits) * 2 / 3) : 0) && // Ensure denominator > 0
                              child_limit + passed_off + completed_visits < collision_limit - params_.GetMinimumRemainingWorkSizeForPicking()) {

                               Node::Iterator child_iter = node->Edges();
                               // Advance iterator safely
                               for(int k=0; k<i && child_iter; ++k) ++child_iter;
                               if (!child_iter) continue;

                               Node* child_node = child_iter.GetOrSpawnNode(node);
                                if (!child_node || child_node->GetN() == 0 || child_node->IsTerminal() || (proven_state_enabled && (child_node->is_known_win.load(std::memory_order_relaxed) || child_node->is_known_loss.load(std::memory_order_relaxed)))) continue;

                               bool passed = false;
                               {
                                   Mutex::Lock lock(picking_tasks_mutex_);
                                   if (picking_tasks_.size() < MAX_TASKS) {
                                       if(child_iter) {
                                           moves_to_path.push_back(child_iter.GetMove());
                                           picking_tasks_.emplace_back(
                                               child_node, static_cast<uint16_t>(current_path.size() + base_depth),
                                               moves_to_path, child_limit);
                                           moves_to_path.pop_back();
                                           task_count_.fetch_add(1, std::memory_order_acq_rel);
                                           task_added_.notify_all();
                                           passed = true;
                                           passed_off += child_limit;
                                       }
                                   }
                               }
                               if (passed) {
                                   (*visits_to_perform.back())[i] = 0;
                               }
                          }
                      }
                 }
             }
        } // End if current_path.back() == -1

        // Select next child logic
        int min_idx = current_path.back();
        bool found_child = false;
        if (!visits_to_perform.empty() && !vtp_last_filled.empty() && vtp_last_filled.back() > min_idx && node && node->HasChildren()) {
            int idx = -1;
            for (auto& child_iter : node->Edges()) {
                idx++;
                if (!child_iter) continue; // Skip if iterator is invalid
                if (idx > min_idx && idx <= vtp_last_filled.back() && idx < 256 && (*visits_to_perform.back())[idx] > 0) {
                    moves_to_path.push_back(child_iter.GetMove());
                    current_path.back() = idx;
                    current_path.push_back(-1);
                    node = child_iter.GetOrSpawnNode(node);
                    found_child = true;
                    break;
                }
                if (idx >= vtp_last_filled.back()) break;
            }
        }

        if (!found_child) {
            node = node ? node->GetParent() : nullptr;
            if (!moves_to_path.empty()) moves_to_path.pop_back();
            current_path.pop_back();
            if (!visits_to_perform.empty()) {
                vtp_buffer.push_back(std::move(visits_to_perform.back()));
                visits_to_perform.pop_back();
                vtp_last_filled.pop_back();
            }
        }

    } // End while current_path.size() > 0
}


void SearchWorker::ExtendNode(Node* node, int depth,
                              const std::vector<Move>& moves_to_node,
                              PositionHistory* history) {
    if (!node || !history) return; // Add null checks

    const auto& board = history->Last().GetBoard();
    auto legal_moves = board.GenerateLegalMoves();

    if (legal_moves.empty()) {
      if (board.IsUnderCheck()) {
        node->MakeTerminal(GameResult::BLACK_WON); // Checkmate is loss for player to move
      } else {
        node->MakeTerminal(GameResult::DRAW); // Stalemate
      }
      return;
    }

    if (node != search_->root_node_) {
      if (!board.HasSufficientMaterial()) {
        node->MakeTerminal(GameResult::DRAW);
        return;
      }
      if (history->Last().GetRule50Ply() >= 100) {
        node->MakeTerminal(GameResult::DRAW);
        return;
      }
      const auto repetitions = history->Last().GetRepetitions();
      if (repetitions >= 2) {
        node->MakeTerminal(GameResult::DRAW);
        return;
      }

      if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
          board.castlings().no_legal_castle() &&
          history->Last().GetRule50Ply() == 0 &&
          (board.ours() | board.theirs()).count() <=
              search_->syzygy_tb_->max_cardinality()) {
        ProbeState state;
        const WDLScore wdl =
            search_->syzygy_tb_->probe_wdl(history->Last(), &state);
        if (state != FAIL) {
          float m = 0.0f;
          {
            SharedMutex::SharedLock lock(search_->nodes_mutex_);
            auto parent = node->GetParent();
            if (parent) {
              m = std::max(0.0f, parent->GetM() - 1.0f);
            }
          }
          if (wdl == WDL_WIN) {
            node->MakeTerminal(GameResult::WHITE_WON, m, Node::Terminal::Tablebase);
          } else if (wdl == WDL_LOSS) {
            node->MakeTerminal(GameResult::BLACK_WON, m, Node::Terminal::Tablebase);
          } else {
            node->MakeTerminal(GameResult::DRAW, m, Node::Terminal::Tablebase);
          }
          search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
          return;
        }
      }
    } // End if not root node

    node->CreateEdges(legal_moves); // Create edges only if not terminal
}


bool SearchWorker::AddNodeToComputation(Node* node) {
    if (!node) return false; // Add null check

    std::vector<Move> moves;
    PositionHistory node_history = search_->GetPositionHistoryAtNode(node); // Reconstruct history

    if (node->HasChildren()) {
        moves.reserve(node->GetNumEdges());
        for (const auto& edge : node->Edges()) {
             if (edge) moves.emplace_back(edge.GetMove());
        }
    } else {
        // Need to check if history is valid before generating moves
        if (node_history.GetLength() == 0) {
             LOGFILE << "Warning: Cannot generate moves, invalid history in AddNodeToComputation.";
             return false; // Cannot proceed without valid history
        }
        moves = node_history.Last().GetBoard().GenerateLegalMoves(); // Use correct history's board
    }

    // Check if history is valid before using
    if (node_history.GetLength() == 0) {
        // Handle error: history reconstruction failed
        LOGFILE << "Warning: Invalid history in AddNodeToComputation.";
        return false;
    }

    return computation_->AddInput(EvalPosition{node_history.GetPositions(), moves},
                                  EvalResultPtr{}) ==
           BackendComputation::FETCHED_IMMEDIATELY;
}


void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}

void SearchWorker::MaybePrefetchIntoCache() {
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->UsedBatchSize() > 0 &&
      static_cast<int>(computation_->UsedBatchSize()) <
          params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->UsedBatchSize(), false);
  }
}

int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0 || !node) return 0; // Add null check

  // --- Check for proven state ---
  if (params_.GetProvenStateHandling() &&
      (node->is_known_win.load(std::memory_order_relaxed) || node->is_known_loss.load(std::memory_order_relaxed))) {
      return 0;
  }
  // --- End Proven State Check ---

  // We are in a leaf, which is not yet being processed.
  if (node->GetNStarted() == 0) {
      if (AddNodeToComputation(node)) { // Node exists here due to prior check
          // Return 1 here, regardless of immediate fetch, as budget was consumed for request
          return 1;
      }
      // If AddNodeToComputation failed (e.g., invalid history), don't count budget.
      // However, the current logic returns 1 even if AddNodeToComputation returns false but doesn't throw.
      // Let's stick to the original logic's return value for now.
      return 1;
  }

  // n = 0 and n_in_flight_ > 0 means node is being extended.
  if (node->GetN() == 0) return 0;
  // Don't prefetch terminal nodes.
  if (node->IsTerminal()) return 0;
  // Don't prefetch if node has no children edges allocated yet.
  if (!node->HasChildren()) return 0;

  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct =
      ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu =
      GetFpu(params_, node, node == search_->root_node_, draw_score);

  for (auto& edge : node->Edges()) {
    if (!edge) continue; // Skip null iterators
    if (edge.GetP() == 0.0f) continue;

    // --- Skip proven loss moves ---
    if (params_.GetProvenStateHandling() && edge.node() && edge.node()->is_known_win.load(std::memory_order_relaxed)) {
        continue;
    }
    // --- End Skip ---

    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu, draw_score),
                        edge);
  }

  if (scores.empty()) return 0; // Return if no eligible children

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;

  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    if (first_unsorted_index < scores.size() && // Ensure not out of bounds
        i + 2 >= first_unsorted_index) {
      const size_t new_unsorted_index = // Use size_t
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      if (first_unsorted_index < new_unsorted_index) {
          std::partial_sort(scores.begin() + first_unsorted_index,
                            scores.begin() + new_unsorted_index, scores.end(),
                            [](const ScoredEdge& a, const ScoredEdge& b) {
                              return a.first < b.first;
                            });
      }
      first_unsorted_index = new_unsorted_index;
    }

     // Ensure index 'i' is valid before accessing scores[i]
     if (i >= scores.size()) break;

    auto edge = scores[i].second;
    if (!edge) continue; // Skip if edge became invalid (shouldn't happen here)

    if (i != scores.size() - 1) {
        // Ensure i+1 is valid
        if (i + 1 >= scores.size()) {
             budget_to_spend = budget; // Last element gets remaining budget
        } else {
             const float next_score = -scores[i + 1].first;
             const float q = edge.GetQ(fpu, draw_score);
              float denom = next_score - q;
             if (denom > 1e-9f) { // Avoid division by zero/small numbers
                 budget_to_spend =
                     std::min(budget, static_cast<int>(std::max(0.0f, edge.GetP() * puct_mult / denom -
                                          edge.GetNStarted())) +
                                          1);
             } else {
                 budget_to_spend = budget;
             }
        }
    } else {
        budget_to_spend = budget;
    }
    budget_to_spend = std::max(0, budget_to_spend); // Ensure non-negative

    // Check if history length exceeds a reasonable limit before appending
    if (history_.GetLength() < 512) { // Example limit
        history_.Append(edge.GetMove());
        const int budget_spent =
            PrefetchIntoCache(edge.GetOrSpawnNode(node), budget_to_spend, !is_odd_depth); // Use GetOrSpawnNode
        history_.Pop();
        budget -= budget_spent;
        total_budget_spent += budget_spent;
    } else {
        // Handle history too long case (e.g., log warning, stop prefetching down this path)
        LOGFILE << "Warning: History too long (" << history_.GetLength() << ") in PrefetchIntoCache, stopping recursion.";
        break;
    }
  }
  return total_budget_spent;
}


void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
}

void SearchWorker::FetchMinibatchResults() {
  for (auto& node_to_process : minibatch_) {
    // Check if node exists before fetching results
    if (node_to_process.node) {
        FetchSingleNodeResult(&node_to_process);
    }
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process) {
  if (!node_to_process || node_to_process->IsCollision() || !node_to_process->node) return; // Null checks

  Node* node = node_to_process->node;
  // Ensure eval pointer is valid before dereferencing
  if (!node_to_process->eval) {
      LOGFILE << "Error: node_to_process->eval is null in FetchSingleNodeResult.";
      return; // Cannot proceed without eval results container
  }

  if (!node_to_process->nn_queried) {
    node_to_process->eval->q = node->GetWL(); // Use WL for terminal Q
    node_to_process->eval->d = node->GetD();
    node_to_process->eval->m = node->GetM();
    return;
  }
  // NN result value is from perspective of player moving INTO node, flip for node's value
  node_to_process->eval->q = -node_to_process->eval->q;

  if (params_.GetWDLRescaleRatio() != 1.0f ||
      (params_.GetWDLRescaleDiff() != 0.0f &&
       search_->contempt_mode_ != ContemptMode::NONE)) {
    bool root_stm = (search_->contempt_mode_ == ContemptMode::BLACK) ==
                    search_->played_history_.Last().IsBlackToMove();
    // Perspective for WDL Rescale depends on whose turn it is at the *current* node
    auto sign = (root_stm ^ (node_to_process->depth % 2 != 0)) ? 1.0f : -1.0f;
    WDLRescale(node_to_process->eval->q, node_to_process->eval->d,
               params_.GetWDLRescaleRatio(),
               search_->contempt_mode_ == ContemptMode::NONE
                   ? 0
                   : params_.GetWDLRescaleDiff(),
               sign, false, params_.GetWDLMaxS());
  }

  if(node->HasChildren()) {
      size_t p_idx = 0;
      // Check policy vector size BEFORE iterating edges
      size_t policy_size = node_to_process->eval->p.size();
      if (policy_size != static_cast<size_t>(node->GetNumEdges()) && policy_size != 0) {
           LOGFILE << "Warning: Policy size mismatch (" << policy_size << ") vs num edges (" << node->GetNumEdges() << ") for node " << node->GetHash();
           // Handle mismatch: maybe zero out policy? Resize? For now, proceed cautiously.
      }

      for (auto& edge : node->Edges()) {
          if (edge) { // Check iterator validity
              if (p_idx < policy_size) {
                 edge.edge()->SetP(node_to_process->eval->p[p_idx]);
              } else {
                 edge.edge()->SetP(0.0f); // Set zero if policy missing
              }
              p_idx++;
          }
      }
       if (params_.GetNoiseEpsilon() > 1e-6f && node == search_->root_node_) { // Use tolerance
         ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                             params_.GetNoiseAlpha());
       }
       node->SortEdges();
  } else if (!node_to_process->eval->p.empty()) {
      // Node has no children, but NN returned policy? Log warning.
      LOGFILE << "Warning: Received policy for node " << node->GetHash() << " which has no children.";
  }
}


void SearchWorker::DoBackupUpdate() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    if(node_to_process.node) { // Add null check
        DoBackupUpdateSingleNode(node_to_process);
    }
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (!node || node_to_process.IsCollision()) { // Add null check and collision check
    return;
  }

  // Ensure eval pointer is valid
  if (!node_to_process.eval) {
      LOGFILE << "Error: node_to_process.eval is null in DoBackupUpdateSingleNode for node " << node->GetHash();
      // If we cannot get eval results, we cannot properly update.
      // Maybe cancel score update? For now, just return.
      node->CancelScoreUpdate(node_to_process.multivisit); // Cancel to decrement n_in_flight
      return;
  }

  // --- Step 1: Check if already proven (NEW) ---
  const bool proven_state_enabled = params_.GetProvenStateHandling();
  if (proven_state_enabled &&
      (node->is_known_win.load(std::memory_order_relaxed) ||
       node->is_known_loss.load(std::memory_order_relaxed))) {
       // Finalize score update only to adjust n_in_flight correctly
       // Use node's current values as dummy inputs since they won't be used for averaging
       node->FinalizeScoreUpdate(node->GetValue(), node->GetD(), node->GetM(), node_to_process.multivisit);
       // Update global stats even if backup stops early
       search_->total_playouts_ += node_to_process.multivisit;
       search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
       search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
       return;
  }

  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && node->GetN()==0; // Check N before update

  // Use Value type (double) for v
  Value v = node_to_process.eval->q;
  float d = node_to_process.eval->d;
  float m = node_to_process.eval->m;
  int n_to_fix = 0;
  Value v_delta = 0.0; // Use Value type
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  uint32_t solid_threshold =
      static_cast<uint32_t>(params_.GetSolidTreeThreshold());

  for (Node *n = node, *p = nullptr; ; n = p) {
    if (!n) break; // Safety break if node becomes null
    p = n->GetParent();

     // --- Re-Check Proven State Before Update ---
     if (proven_state_enabled &&
         (n->is_known_win.load(std::memory_order_relaxed) ||
          n->is_known_loss.load(std::memory_order_relaxed))) {
          // If proven now, just finalize the score update for n_in_flight and stop backup here
          if (n == node) {
             n->FinalizeScoreUpdate(n->GetValue(), n->GetD(), n->GetM(), node_to_process.multivisit);
          } else {
              // If it's an ancestor, it might still need its n_in_flight decremented
              // This requires careful thought - does CancelScoreUpdate handle this?
              // For now, just break to avoid complex state management.
              // Need to ensure n_in_flight is decremented for this path.
              // The simplest is to let the loop continue but use the proven value.
               v = n->is_known_win.load(std::memory_order_relaxed) ? kValueKnownWin : kValueKnownLoss;
               d = 0.0f;
               // M value for proven nodes isn't strictly defined here, use current M? Or recalculate?
               // Let's use current M, as it reflects path length so far.
               m = n->GetM();
               n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit); // Finalize this node
               // Continue propagating the *proven* value up
          }
         // Don't break here anymore, let the proven value propagate up
     }

    // Get current terminal/known values before update if needed
    bool n_is_terminal_before_update = n->IsTerminal();
    Value n_value_before_update = n->GetValue();
    float n_d_before_update = n->GetD();
    float n_m_before_update = n->GetM();

    // Update node 'n'
    if (n_is_terminal_before_update) {
        // If already terminal, use its terminal value for propagation
        v = n_value_before_update;
        d = n_d_before_update;
        m = n_m_before_update;
        // Finalize score update is still needed to handle n_in_flight correctly
        n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    } else if (n == node || n_to_fix > 0) {
        if (n == node) { // Original leaf node update
            n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
        } else { // Parent node adjustment due to child becoming terminal
            n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
        }
        n_to_fix = 0;
    } else {
        // Standard backup for ancestor nodes
         n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    }

    // --- Proven State Check on Parent 'p' ---
    if (proven_state_enabled && p != nullptr && !p->IsTerminal() &&
        !p->is_known_win.load(std::memory_order_relaxed) && !p->is_known_loss.load(std::memory_order_relaxed) ) {
        Value min_bound = p->GetMinValue();
        Value max_bound = p->GetMaxValue();

        // Check bounds AFTER the update of 'p' (which happens in the next iteration or AdjustForTerminal)
        // This check should ideally happen after 'p' has incorporated the information from 'n'.
        // Moving this check to AFTER the MaybeSetBounds call, or integrate it there.
        // For now, keep it here, understanding it might be slightly delayed.
        if (max_bound >= kValueKnownWin) {
            p->is_known_win.store(true, std::memory_order_relaxed);
        } else if (min_bound <= kValueKnownLoss) {
            p->is_known_loss.store(true, std::memory_order_relaxed);
        }
    }
    // --- End Proven State Check ---

    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_, 0);
      }
    }

    if (!p) break; // Reached root's parent (null)

    bool old_update_parent_bounds = update_parent_bounds;
    // Check parent terminal/known state
    bool parent_is_final = p->IsTerminal() || (proven_state_enabled && (p->is_known_win.load(std::memory_order_relaxed) || p->is_known_loss.load(std::memory_order_relaxed)));

    if (parent_is_final) {
        n_to_fix = 0; // Don't adjust final parents
        update_parent_bounds = false; // Don't try to set bounds on final parents
    } else {
        // Try setting parent bounds (only if enabled and parent not final)
        update_parent_bounds =
            old_update_parent_bounds && p != search_->root_node_ &&
            MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);
    }


    // Flip perspective for next level up
    v = -v;
    v_delta = -v_delta;
    m++; // Increment M for distance

    // Update best move if parent is root
    if (p == search_->root_node_) {
         bool n_became_final = !n_is_terminal_before_update && (n->IsTerminal() || (proven_state_enabled && (n->is_known_win.load(std::memory_order_relaxed) || n->is_known_loss.load(std::memory_order_relaxed))));
         // Update best edge if:
         // 1. Child just became terminal/proven OR
         // 2. Child is not the current best and its N is now >= current best's N
         if ((old_update_parent_bounds && n_became_final) ||
             (search_->current_best_edge_.node() != n && // Compare nodes, not edges directly
              search_->current_best_edge_.GetN() <= n->GetN())) {
           search_->current_best_edge_ =
               search_->GetBestChildNoTemperature(search_->root_node_, 0);
         }
    }
  } // End for loop propagating backup

  // Update global counters outside the loop
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}


bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  Value* v_delta, float* d_delta,
                                  float* m_delta) {
    if (!p) return false; // Null check

    // This function now implicitly uses GetMinValue/GetMaxValue via the proven state flags set in Backup
    // We only need to handle the StickyEndgames logic here if the node *just* became terminal via game rules

    bool bounds_changed = false;
    auto [original_lower, original_upper] = p->GetBounds(); // Get tuple directly

    // Check if bounds can be updated based on children's known/terminal states
    // (This logic might be redundant if flags are set reliably in Backup, but kept for StickyEndgames)
    GameResult new_lower = GameResult::BLACK_WON;
    GameResult new_upper = GameResult::WHITE_WON; // Start with widest bounds

    if (p->HasChildren()) {
         bool can_force_win = false;
         bool can_avoid_loss = false; // Can parent achieve at least a draw?
         bool can_avoid_win = false; // Can opponent force draw or win for themselves?
         bool all_children_loss_or_draw = true; // Assume loss/draw unless win found
         bool all_children_win_or_draw = true; // Assume win/draw unless loss found

         for (const auto& edge : p->Edges()) {
             if (!edge) continue;
             Node* child = edge.node();
             GameResult child_lower, child_upper;

             if (child && child->is_known_win.load(std::memory_order_relaxed)){
                 child_lower = GameResult::WHITE_WON; child_upper = GameResult::WHITE_WON;
             } else if (child && child->is_known_loss.load(std::memory_order_relaxed)) {
                  child_lower = GameResult::BLACK_WON; child_upper = GameResult::BLACK_WON;
             } else if (child && child->IsTerminal()) {
                 // Get bounds from terminal result (already set correctly)
                 auto [cl, cu] = child->GetBounds();
                 child_lower = cl; child_upper = cu;
             } else if (child){
                 // Get bounds from non-terminal child
                 auto [cl, cu] = child->GetBounds();
                 child_lower = cl; child_upper = cu;
             } else {
                 // Dangling edge - assume regular bounds? Or skip? Assume regular for now.
                 child_lower = GameResult::BLACK_WON;
                 child_upper = GameResult::WHITE_WON;
             }

             // Convert child bounds to parent perspective (flip and swap)
             GameResult parent_view_lower = FlipGameResult(child_upper);
             GameResult parent_view_upper = FlipGameResult(child_lower);

             // Update parent's potential best/worst outcomes
             new_lower = std::max(new_lower, parent_view_lower); // Parent's best outcome = max(children's worst for opponent)
             new_upper = std::min(new_upper, parent_view_upper); // Parent's worst outcome = min(children's best for opponent)

             // Track possibilities for quicker bound setting
             if (parent_view_lower >= GameResult::DRAW) can_avoid_loss = true;
             if (parent_view_upper <= GameResult::DRAW) can_avoid_win = true; // Check if opponent can force draw/loss
             if (parent_view_lower == GameResult::WHITE_WON) can_force_win = true;

             // Update flags for checking all children status
             if (parent_view_lower < GameResult::DRAW) all_children_win_or_draw = false;
             if (parent_view_upper > GameResult::DRAW) all_children_loss_or_draw = false;

             // Early exit if bounds become fully open again
             if (new_lower == GameResult::BLACK_WON && new_upper == GameResult::WHITE_WON) break;
         } // End for loop children

        // Final bound determination based on aggregated results
         if (can_force_win) { // If any child guarantees win for parent
             new_lower = GameResult::WHITE_WON;
             new_upper = GameResult::WHITE_WON;
         } else if (can_avoid_win) { // If opponent can force draw or win for themselves (loss/draw for parent)
             new_upper = GameResult::DRAW;
             // Lower bound remains max of children's worst outcomes
         }

         if (!can_avoid_loss) { // If all moves lead to loss for parent
             new_lower = GameResult::BLACK_WON;
             new_upper = GameResult::BLACK_WON;
         } else if (!can_force_win && all_children_loss_or_draw) { // Can avoid loss, cannot force win, all lead to draw/loss
             new_lower = GameResult::BLACK_WON; // Worst outcome is loss
             new_upper = GameResult::DRAW;      // Best outcome is draw
         } else if (can_avoid_loss && all_children_win_or_draw && !can_force_win) { // Can avoid loss, can potentially win, all moves are win/draw
             new_lower = GameResult::DRAW;      // Worst outcome is draw
             new_upper = GameResult::WHITE_WON; // Best outcome is win
         }
         // Case: can_avoid_loss=true, can_force_win=false, some children lead to win, some lead to loss/draw
         // -> new_lower = BLACK_WON, new_upper = WHITE_WON (or DRAW if can_avoid_win=true)
         // This is handled by the initial max/min updates unless overridden above.

         // Clamp bounds
         new_lower = std::min(new_upper, new_lower); // Ensure lower <= upper

    } else {
         // Node has no children yet, bounds remain default unknown
         return false;
    }


    // Check if bounds actually changed
    if (new_lower != original_lower || new_upper != original_upper) {
         p->SetBounds(new_lower, new_upper);
         bounds_changed = true;

         // Check for proven state flags based on new bounds
         if (params_.GetProvenStateHandling()) {
             if (new_upper >= kValueKnownWin && !p->is_known_win.load(std::memory_order_relaxed)) {
                 p->is_known_win.store(true, std::memory_order_relaxed);
             }
             if (new_lower <= kValueKnownLoss && !p->is_known_loss.load(std::memory_order_relaxed)) {
                 p->is_known_loss.store(true, std::memory_order_relaxed);
             }
         }
    }

    // If bounds collapsed to a terminal state
    if (new_lower == new_upper && !p->IsTerminal()) { // Check IsTerminal again
        *n_to_fix = p->GetN();
        if (*n_to_fix <= 0) {
            // Cannot make terminal if N=0, as it hasn't been visited properly.
            // This might happen if bounds collapse due to unvisited children being proven.
            // Revert the bound change? Or just don't make terminal yet?
            // Let's not make it terminal yet, but keep the bounds.
             LOGFILE << "Warning: Bounds collapsed for node " << p->GetHash() << " with N=0, not making terminal.";
             return bounds_changed; // Return whether bounds changed, but don't make terminal.
        }

        Value cur_v = p->GetValue(); // Use Value
        float cur_d = p->GetD();
        float cur_m = p->GetM();

        // Find if any child was a TB win to set the type correctly
        bool prefer_tb = false;
        if(p->HasChildren()){
            for(const auto& edge : p->Edges()){
                 if(edge && edge.IsTbTerminal()) { prefer_tb = true; break;} // Check edge validity
            }
        }

        // Make terminal with the proven result
        // Estimate M based on passed child M + 1
        p->MakeTerminal(
            new_lower,
            m + 1.0f, // Use child's M + 1 (passed in as 'm')
            prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);

        // Calculate deltas for parent adjustment
        *v_delta = -(p->GetValue() - cur_v); // Delta is negated for parent perspective, use Value
        *d_delta = p->GetD() - cur_d;
        *m_delta = p->GetM() - cur_m;

        return true; // Indicate bounds were set and collapsed
    }

    return bounds_changed; // Indicate if bounds were updated
}




void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace classic
}  // namespace lczero
