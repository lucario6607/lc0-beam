/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header remains the same) ...
*/

#pragma once

#include <array>
#include <condition_variable>
#include <functional>
#include <limits> // Required for numeric_limits
#include <optional>
#include <shared_mutex>
#include <thread>

// Corrected Includes
#include "callbacks.h" // Contains ThinkingInfo etc.
#include "chess.h"     // <<< CORRECTED PATH: Defines Value, kValueMate etc.
#include "position.h" // <<< CORRECTED PATH: Defines PositionHash
#include "uciloop.h"   // Contains UciResponder
#include "neural/backend.h"
#include "search/classic/node.h" // Include the corrected node.h
#include "search/classic/params.h"
#include "search/classic/stoppers/timemgr.h"
#include "syzygy/syzygy.h" // May need adjustment based on actual location
#include "utils/logging.h"
#include "utils/mutex.h"

namespace lczero {
namespace classic {

// Define constants for known win/loss based on LC0's Mate value
constexpr Value kValueKnownWin = kValueMate;
constexpr Value kValueKnownLoss = -kValueMate;

// NOTE: TTEntry structure definition would go here or be included if separate
// Example placeholder:
struct TTEntry {
    // ... existing fields like policy_data, value, visits, age, etc. ...
    bool known_win = false;
    bool known_loss = false;
};


class Search {
 public:
  Search(const NodeTree& tree, Backend* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite, bool ponder,
         const OptionsDict& options, SyzygyTablebase* syzygy_tb);

  ~Search();

  void StartThreads(size_t how_many);
  void RunBlocking(size_t threads);
  void Stop();
  void Abort();
  void Wait();
  bool IsSearchActive() const;

  std::pair<Move, Move> GetBestMove();
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  std::int64_t GetTotalPlayouts() const;
  const SearchParams& GetParams() const { return params_; }
  void ResetBestMove();
  std::optional<EvalResult> GetCachedNNEval(const Node* node) const;

 private:
  void EnsureBestMoveKnown();
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo();
  void FireStopInternal();
  void SendMovesStats() const;
  void WatchdogThread();
  void PopulateCommonIterationStats(IterationStats* stats);
  std::vector<std::string> GetVerboseStats(Node* node) const;
  float GetDrawScore(bool is_odd_depth) const;
  void CancelSharedCollisions();
  PositionHistory GetPositionHistoryAtNode(const Node* node) const;

  // NOTE: Placeholder for StoreTT declaration
  void StoreTT(PositionHash hash, Node* node); // PositionHash should now be defined


  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  std::atomic<bool> stop_{false};
  std::condition_variable watchdog_cv_;
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<SearchStopper> stopper_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  SyzygyTablebase* syzygy_tb_;
  const PositionHistory& played_history_;

  Backend* const backend_;
  BackendAttributes backend_attributes_;
  const SearchParams params_;
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  bool root_is_in_dtz_ = false;
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<UciResponder> uci_responder_;
  ContemptMode contempt_mode_;
  friend class SearchWorker;
};

// Single thread worker... (SearchWorker class remains the same as previous correct version) ...
class SearchWorker {
 // ... (Contents same as previous correct version) ...
};


}  // namespace classic
}  // namespace lczero
