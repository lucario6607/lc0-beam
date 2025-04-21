/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header remains the same) ...
*/

#pragma once

#include <array>
#include <condition_variable>
#include <functional>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <thread>

// Corrected Includes (NO chess/ prefix - Final attempt based on build flags)
#include "callbacks.h" // <<< Final Path Correction
#include "chess.h"     // <<< Final Path Correction
#include "position.h" // <<< Final Path Correction
#include "uciloop.h"   // <<< Final Path Correction
#include "gamestate.h" // <<< Final Path Correction
#include "neural/backend.h"
#include "search/classic/node.h" // Includes node.h (which uses direct includes)
#include "search/classic/params.h"
#include "search/classic/stoppers/timemgr.h"
#include "syzygy/syzygy.h" // Path may vary
#include "utils/logging.h"
#include "utils/mutex.h"


namespace lczero {
namespace classic {

// Define constants for known win/loss based on LC0's Mate value
constexpr Value kValueKnownWin = kValueMate;
constexpr Value kValueKnownLoss = -kValueMate;

// TTEntry structure placeholder (assuming it's defined elsewhere or implicitly)
// Example:
// struct TTEntry {
//     // ... existing fields like policy_data, value, visits, age, etc. ...
//     bool known_win = false;
//     bool known_loss = false;
// };


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

// --- SearchWorker class ---
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params);
  ~SearchWorker();

  void RunBlocking();
  void ExecuteOneIteration();
  void InitializeIteration(std::unique_ptr<BackendComputation> computation);
  void GatherMinibatch();
  void CollectCollisions();
  void MaybePrefetchIntoCache();
  void RunNNComputation();
  void FetchMinibatchResults();
  void DoBackupUpdate();
  void UpdateCounters();

 private:
  struct NodeToProcess;
  struct TaskWorkspace;
  struct PickTask;

  struct NodeToProcess {
    bool IsExtendable() const { return node && !is_collision && !node->IsTerminal(); }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return node && (is_cache_hit || node->IsTerminal());
    }

    Node* node;
    std::unique_ptr<EvalResult> eval;
    int multivisit = 0;
    int maxvisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;
    std::vector<Move> moves_to_visit;
    bool ooo_completed = false;

    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count, 0);
    }
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count, int max_count) {
      return NodeToProcess(node, depth, true, collision_count, max_count);
    }
    static NodeToProcess Visit(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1, 0);
    }

   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit,
                  int max_count)
        : node(node),
          eval(std::make_unique<EvalResult>()),
          multivisit(multivisit),
          maxvisit(max_count),
          depth(depth),
          is_collision(is_collision) {}
  };

  struct TaskWorkspace {
    std::array<Node::Iterator, 256> cur_iters;
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    std::vector<Move> moves_to_path;
    PositionHistory history;
    TaskWorkspace();
  };

  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;
    Node* start;
    int base_depth;
    int collision_limit;
    std::vector<Move> moves_to_base;
    std::vector<NodeToProcess> results;
    int start_idx;
    int end_idx;
    bool complete = false;

    PickTask(Node* node, uint16_t depth, const std::vector<Move>& base_moves,
             int collision_limit);
    PickTask(int start_idx, int end_idx);
  };

  bool AddNodeToComputation(Node* node);
  int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix, Value* v_delta,
                      float* d_delta, float* m_delta);
  void PickNodesToExtend(int collision_limit);
  void PickNodesToExtendTask(Node* starting_point, int base_depth,
                             int collision_limit,
                             const std::vector<Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace);
  void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
  void ProcessPickedTask(int batch_start, int batch_end,
                         TaskWorkspace* workspace);
  void ExtendNode(Node* node, int depth, const std::vector<Move>& moves_to_add,
                  PositionHistory* history);
  void FetchSingleNodeResult(NodeToProcess* node_to_process);
  void RunTasks(int tid);
  void ResetTasks();
  int WaitForTasks();

  Search* const search_;
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<BackendComputation> computation_;
  int task_workers_;
  int target_minibatch_size_;
  int max_out_of_order_;
  PositionHistory history_;
  int number_out_of_order_ = 0;
  const SearchParams& params_;
  std::unique_ptr<Node> precached_node_;
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;

  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_;
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_;
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;
};


}  // namespace classic
}  // namespace lczero
