/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#pragma once

// --- Core Type Definitions FIRST ---
#include "chess/types.h"        // Defines lczero::Value, lczero::Move, lczero::Eval, lczero::MoveList, lczero::GameResult, kValueMate etc.

// --- Standard Library Includes ---
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>
#include <utility>

// --- Other Necessary LC0 Includes ---
#include "chess/position.h"     // Defines lczero::PositionHash, lczero::Position
#include "chess/uciloop.h"      // Defines lczero::UciResponder, lczero::ThinkingInfo, lczero::BestMoveInfo
#include "chess/gamestate.h"    // Defines lczero::PositionHistory
#include "neural/backend.h"     // Defines lczero::Backend, lczero::BackendAttributes, lczero::BackendComputation
#include "proto/net.pb.h"       // Defines EvalResult
#include "search/classic/node.h" // Includes node.h (which includes its own dependencies like types.h)
#include "search/classic/params.h" // Defines classic::SearchParams, classic::ContemptMode
#include "search/classic/stoppers/timemgr.h" // Defines lczero::SearchStopper, lczero::StoppersHints
#include "syzygy/syzygy.h"      // Defines lczero::SyzygyTablebase
#include "utils/optionsdict.h"  // Defines lczero::OptionsDict
#include "utils/mutex.h"        // Defines Mutex, SharedMutex
#include "utils/logging.h"      // Defines LOGFILE


// --- Forward Declarations ---
namespace lczero {
class NodeTree;
namespace classic {
class SearchWorker; // Forward declare worker class
} // namespace classic
} // namespace lczero


namespace lczero {
using IterationStats = classic::IterationStats;
using StoppersHints = classic::StoppersHints;
namespace classic {

// Constants defined using types from lczero namespace
constexpr lczero::Value kValueKnownWin = lczero::kValueMate;
constexpr lczero::Value kValueKnownLoss = -lczero::kValueMate;

class Search {
 public:
  Search(const NodeTree& tree, lczero::Backend* network,
         std::unique_ptr<lczero::UciResponder> uci_responder,
         const lczero::MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<lczero::classic::SearchStopper> stopper,
         bool infinite, bool ponder,
         const lczero::OptionsDict& options, lczero::SyzygyTablebase* syzygy_tb);

  ~Search();

  void StartThreads(size_t how_many);
  void RunBlocking(size_t threads);
  void Stop();
  void Abort();
  void Wait();
  bool IsSearchActive() const;

  std::pair<lczero::Move, lczero::Move> GetBestMove();
  lczero::Eval GetBestEval(lczero::Move* move = nullptr, bool* is_terminal = nullptr) const; // <<< Uses lczero::Eval
  std::int64_t GetTotalPlayouts() const;
  const SearchParams& GetParams() const { return params_; }
  void ResetBestMove();
  std::optional<EvalResult> GetCachedNNEval(const Node* node) const; // <<< Uses EvalResult

 private:
  void EnsureBestMoveKnown();
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_);
  void MaybeTriggerStop(const lczero::IterationStats& stats, lczero::StoppersHints* hints); // <<< Uses lczero::IterationStats, lczero::StoppersHints
  void MaybeOutputInfo();
  void SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_);
  void FireStopInternal();
  void SendMovesStats() const REQUIRES(counters_mutex_);
  void WatchdogThread();
  void PopulateCommonIterationStats(lczero::IterationStats* stats); // <<< Uses lczero::IterationStats
  std::vector<std::string> GetVerboseStats(Node* node) const;
  float GetDrawScore(bool is_odd_depth) const;
  void CancelSharedCollisions() REQUIRES(nodes_mutex_);
  lczero::PositionHistory GetPositionHistoryAtNode(const Node* node) const; // <<< Uses lczero::PositionHistory

  // void StoreTT(uint64_t hash, Node* node); // <<< Uses lczero::PositionHash (Commented out as likely unused/misplaced)


  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  std::atomic<bool> stop_{false};
  std::condition_variable watchdog_cv_;
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  lczero::Move final_bestmove_ GUARDED_BY(counters_mutex_); // <<< Uses lczero::Move
  lczero::Move final_pondermove_ GUARDED_BY(counters_mutex_); // <<< Uses lczero::Move
  std::unique_ptr<lczero::classic::SearchStopper> stopper_ GUARDED_BY(counters_mutex_); // <<< Uses lczero::SearchStopper

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_; // Managed by NodeTree, non-owning pointer
  lczero::SyzygyTablebase* syzygy_tb_; // Non-owning pointer
  const lczero::PositionHistory& played_history_; // <<< Uses lczero::PositionHistory

  lczero::Backend* const backend_;
  lczero::BackendAttributes backend_attributes_;
  const SearchParams params_;
  const lczero::MoveList searchmoves_; // <<< Uses lczero::MoveList
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  bool root_is_in_dtz_ = false;
  std::atomic<int> tb_hits_{0};
  const lczero::MoveList root_move_filter_; // <<< Uses lczero::MoveList

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  lczero::ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_); // <<< Uses lczero::ThinkingInfo
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  mutable std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<lczero::UciResponder> uci_responder_; // <<< Uses lczero::UciResponder
  ContemptMode contempt_mode_;
  friend class SearchWorker;

  // --- Nested Structs/Classes for SearchWorker (Defined within SearchWorker below) ---
  struct NodeToProcess;
  struct TaskWorkspace;
  struct PickingTask;

   // --- SearchWorker Definition ---
   // Define SearchWorker within Search or forward declare and define later
   // Defining it here allows direct access to Search's private members via friendship
   class SearchWorker {
     public:
       SearchWorker(Search* search, const SearchParams& params);
       ~SearchWorker();
       void RunBlocking();

     private:
       // Nested struct definitions moved here for clarity
       struct NodeToProcess {
           Node* node = nullptr;
           std::shared_ptr<EvalResult> eval; // Use shared_ptr
           int multivisit = 0;
           int maxvisit = 0;
           uint16_t depth = 0;
           bool nn_queried = false;
           bool is_collision = false;
           std::vector<lczero::Move> moves_to_visit; // <<< Uses lczero::Move

           static NodeToProcess Collision(Node* node, uint16_t depth,
                                          int collision_count, int max_count = 0) {
               return NodeToProcess(node, depth, true, collision_count, max_count);
           }
           static NodeToProcess Visit(Node* node, uint16_t depth) {
               return NodeToProcess(node, depth, false, 1, 0);
           }
       private:
           NodeToProcess(Node* n, uint16_t d, bool is_coll, int mv, int max_v)
               : node(n),
                 eval(std::make_shared<EvalResult>()), // Initialize shared_ptr
                 multivisit(mv),
                 maxvisit(max_v),
                 depth(d),
                 is_collision(is_coll) {}
       };

       struct TaskWorkspace {
           std::array<Node::Iterator, 256> cur_iters;
           std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
           std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
           std::vector<int> vtp_last_filled;
           std::vector<int> current_path;
           std::vector<lczero::Move> moves_to_path; // <<< Uses lczero::Move
           lczero::PositionHistory history; // Needs initialization in constructor
           TaskWorkspace(const lczero::PositionHistory& initial_history); // Add constructor
       };

       struct PickingTask {
           Node* node;
           uint16_t base_depth;
           std::vector<lczero::Move> moves; // <<< Uses lczero::Move
           int visits;

           PickingTask(Node* n, uint16_t depth, const std::vector<lczero::Move>& m, int v)
               : node(n), base_depth(depth), moves(m), visits(v) {}
       };

      // Private methods of SearchWorker
      void ExecuteOneIteration();
      void InitializeIteration(std::unique_ptr<lczero::BackendComputation> computation);
      void GatherMinibatch();
      void CollectCollisions();
      void MaybePrefetchIntoCache();
      void RunNNComputation();
      void FetchMinibatchResults();
      void DoBackupUpdate();
      void UpdateCounters();
      bool AddNodeToComputation(Node* node);
      int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
      void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_);
      bool MaybeSetBounds(Node* p, float m, int* n_to_fix, lczero::Value* v_delta, // <<< Uses lczero::Value*
                           float* d_delta, float* m_delta) REQUIRES(search_->nodes_mutex_);
      void PickNodesToExtend(int collision_limit);
      void PickNodesToExtendTask(Node* starting_point, int base_depth,
                                 int collision_limit,
                                 const std::vector<lczero::Move>& moves_to_base, // <<< Uses lczero::Move
                                 std::vector<NodeToProcess>* receiver,
                                 TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS;
      void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
      void ProcessPickedTask(int batch_start, int batch_end,
                             TaskWorkspace* workspace);
      void ExtendNode(Node* node, int depth, const std::vector<lczero::Move>& moves_to_add, // <<< Uses lczero::Move
                      lczero::PositionHistory* history); // <<< Uses lczero::PositionHistory
      void FetchSingleNodeResult(NodeToProcess* node_to_process);
      void RunTasks(int tid);
      void ResetTasks();
      int WaitForTasks();

       // Member variables of SearchWorker
       Search* const search_;
       std::vector<NodeToProcess> minibatch_;
       std::unique_ptr<lczero::BackendComputation> computation_;
       const int task_workers_;
       lczero::PositionHistory history_; // Local copy for worker
       int number_out_of_order_ = 0;
       const SearchParams& params_;
       const bool moves_left_support_;
       lczero::IterationStats iteration_stats_; // <<< Uses lczero::IterationStats
       lczero::StoppersHints latest_time_manager_hints_; // <<< Uses lczero::StoppersHints

       Mutex picking_tasks_mutex_;
       std::vector<PickingTask> picking_tasks_ GUARDED_BY(picking_tasks_mutex_);
       std::atomic<int> task_count_{0};
       std::condition_variable task_added_;
       std::thread picker_thread_; // Only one picker thread
       TaskWorkspace task_workspace_; // Workspace for the picker thread
   }; // End class SearchWorker

   // Search member variables that depend on SearchWorker definition
   lczero::IterationStats iteration_stats_; // <<< Uses lczero::IterationStats
   lczero::StoppersHints latest_time_manager_hints_; // <<< Uses lczero::StoppersHints

}; // End class Search


}  // namespace classic
}  // namespace lczero
