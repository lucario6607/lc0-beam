/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#pragma once

#include <array>
#include <atomic>               // Added for std::atomic
#include <chrono>               // Added for time points
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>               // Added for std::unique_ptr, std::shared_ptr
#include <mutex>                // Added for std::mutex related types (if needed by impl)
#include <optional>
#include <shared_mutex>
#include <string>               // Added for std::string
#include <thread>
#include <vector>
#include <utility>

// Includes needed for types used in this header
#include "chess/types.h"        // Defines lczero::Value, lczero::Move, lczero::Eval, lczero::MoveList, lczero::GameResult, kValueMate etc.
#include "chess/position.h"     // Defines lczero::PositionHash, lczero::Position
#include "chess/uciloop.h"      // Defines lczero::UciResponder, lczero::ThinkingInfo, lczero::BestMoveInfo
#include "chess/gamestate.h"    // Defines lczero::PositionHistory
#include "neural/backend.h"     // Defines lczero::Backend, lczero::BackendAttributes, lczero::BackendComputation
#include "search/classic/node.h" // Defines classic::Node, classic::EdgeAndNode etc. (includes its own dependencies)
#include "search/classic/params.h" // Defines classic::SearchParams, classic::ContemptMode
#include "search/search_stopper.h" // Defines lczero::SearchStopper, lczero::StoppersHints (used directly)
#include "search/stats.h"       // Defines lczero::IterationStats (used directly)
#include "syzygy/syzygy.h"      // Defines lczero::SyzygyTablebase
#include "utils/optionsdict.h"  // Defines lczero::OptionsDict
#include "utils/mutex.h"        // Defines Mutex, SharedMutex
#include "utils/logging.h"      // Defines LOGFILE
#include "proto/net.pb.h"       // Defines EvalResult

// Forward declarations
namespace lczero {
class NodeTree; // Forward declare
namespace classic {
class SearchWorker; // Forward declare worker class
} // namespace classic
} // namespace lczero


namespace lczero {
namespace classic {

// Constants defined using types from lczero namespace
constexpr lczero::Value kValueKnownWin = lczero::kValueMate;
constexpr lczero::Value kValueKnownLoss = -lczero::kValueMate;

// TTEntry structure placeholder
// ...

class Search {
 public:
  Search(const NodeTree& tree, lczero::Backend* network,
         std::unique_ptr<lczero::UciResponder> uci_responder,
         const lczero::MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<lczero::SearchStopper> stopper, // Use lczero::SearchStopper
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
  lczero::Eval GetBestEval(lczero::Move* move = nullptr, bool* is_terminal = nullptr) const;
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
  int64_t GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_); // Added annotation based on impl
  void MaybeTriggerStop(const lczero::IterationStats& stats, lczero::StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_); // Added annotation based on impl
  void FireStopInternal();
  void SendMovesStats() const REQUIRES(counters_mutex_); // Added annotation based on impl
  void WatchdogThread();
  void PopulateCommonIterationStats(lczero::IterationStats* stats);
  std::vector<std::string> GetVerboseStats(Node* node) const;
  float GetDrawScore(bool is_odd_depth) const;
  void CancelSharedCollisions() REQUIRES(nodes_mutex_); // Added annotation based on impl
  lczero::PositionHistory GetPositionHistoryAtNode(const Node* node) const;

  // StoreTT likely belongs elsewhere (e.g., a dedicated TT class) or needs full definition if used here.
  // void StoreTT(lczero::PositionHash hash, Node* node);


  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  std::atomic<bool> stop_{false};
  std::condition_variable watchdog_cv_;
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  lczero::Move final_bestmove_ GUARDED_BY(counters_mutex_);
  lczero::Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<lczero::SearchStopper> stopper_ GUARDED_BY(counters_mutex_); // Use lczero::SearchStopper

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_; // Managed by NodeTree, non-owning pointer
  lczero::SyzygyTablebase* syzygy_tb_; // Non-owning pointer
  const lczero::PositionHistory& played_history_; // Reference to history from NodeTree

  lczero::Backend* const backend_; // Non-owning pointer
  lczero::BackendAttributes backend_attributes_;
  const SearchParams params_;
  const lczero::MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  bool root_is_in_dtz_ = false;
  std::atomic<int> tb_hits_{0};
  const lczero::MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  lczero::ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  // Make nps_start_time_ mutable as GetTimeSinceFirstBatch is const but modifies it
  mutable std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  // std::atomic<int> backend_waiting_counter_{0}; // Seems unused? Comment out or remove if confirmed.
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<lczero::UciResponder> uci_responder_;
  ContemptMode contempt_mode_;
  friend class SearchWorker;
};

// --- SearchWorker class ---
// Needs forward declaration or full definition before usage in Search member functions
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params);
  ~SearchWorker();

  void RunBlocking();

 private:
  // Forward declare nested structs to keep header clean
  struct NodeToProcess;
  struct TaskWorkspace;
  struct PickingTask; // Renamed from PickTask for clarity

  // Public methods called by Search or thread entry points
  void ExecuteOneIteration();
  void InitializeIteration(std::unique_ptr<lczero::BackendComputation> computation);
  void GatherMinibatch();
  void CollectCollisions();
  void MaybePrefetchIntoCache();
  void RunNNComputation();
  void FetchMinibatchResults();
  void DoBackupUpdate();
  void UpdateCounters();

  // Internal helper methods
  bool AddNodeToComputation(Node* node);
  int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_); // Add annotation
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix, lczero::Value* v_delta,
                      float* d_delta, float* m_delta) REQUIRES(search_->nodes_mutex_); // Add annotation
  void PickNodesToExtend(int collision_limit);
  void PickNodesToExtendTask(Node* starting_point, int base_depth,
                             int collision_limit,
                             const std::vector<lczero::Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS; // Already annotated
  void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
  void ProcessPickedTask(int batch_start, int batch_end,
                         TaskWorkspace* workspace);
  void ExtendNode(Node* node, int depth, const std::vector<lczero::Move>& moves_to_add,
                  lczero::PositionHistory* history);
  void FetchSingleNodeResult(NodeToProcess* node_to_process);
  void RunTasks(int tid); // Thread entry point
  void ResetTasks();
  int WaitForTasks();

  // --- Member Variables ---
  Search* const search_; // Non-owning pointer to parent Search object
  std::vector<NodeToProcess> minibatch_; // Nodes selected for processing in this iteration
  std::unique_ptr<lczero::BackendComputation> computation_; // Handle for NN computation batch
  const int task_workers_; // Number of picker threads (if any)
  // int target_minibatch_size_; // Seems unused?
  // int max_out_of_order_; // Seems unused?
  lczero::PositionHistory history_; // Local copy of history for traversing the tree
  int number_out_of_order_ = 0; // Counter for out-of-order processing
  const SearchParams& params_; // Reference to search parameters
  // std::unique_ptr<Node> precached_node_; // Seems unused?
  const bool moves_left_support_; // Does the backend support MLH?
  lczero::IterationStats iteration_stats_; // Stats for the current iteration
  lczero::StoppersHints latest_time_manager_hints_; // Hints for time management

  // --- Task Picking Members (for multi-threaded picking) ---
  Mutex picking_tasks_mutex_;
  std::vector<PickingTask> picking_tasks_ GUARDED_BY(picking_tasks_mutex_); // Tasks for picker threads
  std::atomic<int> task_count_{0}; // Number of available tasks
  // std::atomic<int> task_taking_started_{0}; // Seems unused?
  // std::atomic<int> tasks_taken_{0}; // Seems unused?
  // std::atomic<int> completed_tasks_{0}; // Seems unused?
  std::condition_variable task_added_; // CV for picker threads to wait on
  std::thread picker_thread_; // The picker thread (if task_workers_ > 0)
  // std::vector<std::thread> task_threads_; // Should be just one picker_thread_?
  // std::vector<TaskWorkspace> task_workspaces_; // Should be just one task_workspace_?
  TaskWorkspace task_workspace_; // Workspace for the picker thread
  // bool exiting_ = false; // Use search_->stop_ instead


   // --- Nested Struct Definitions (moved from private for definition) ---
    struct NodeToProcess {
        Node* node = nullptr;
        std::shared_ptr<EvalResult> eval; // Use shared_ptr as it's passed to backend
        int multivisit = 0;
        int maxvisit = 0; // Max visits allowed for this branch (pruning)
        uint16_t depth = 0;
        bool nn_queried = false; // Was NN actually queried for this node?
        // bool is_cache_hit = false; // Handled by nn_queried and eval content
        bool is_collision = false; // Is this a collision entry?
        std::vector<lczero::Move> moves_to_visit; // Path to this node (for history reconstruction)
        // bool ooo_completed = false; // Out-of-order flag seems unused

        // Static factory methods
        static NodeToProcess Collision(Node* node, uint16_t depth,
                                       int collision_count, int max_count = 0) { // Added default for max_count
            return NodeToProcess(node, depth, true, collision_count, max_count);
        }
        static NodeToProcess Visit(Node* node, uint16_t depth) {
            return NodeToProcess(node, depth, false, 1, 0);
        }

    private:
        // Private constructor for factory methods
        NodeToProcess(Node* n, uint16_t d, bool is_coll, int mv, int max_v)
            : node(n),
              // Initialize shared_ptr here
              eval(std::make_shared<EvalResult>()),
              multivisit(mv),
              maxvisit(max_v),
              depth(d),
              is_collision(is_coll) {}
    };


    struct TaskWorkspace {
        std::array<Node::Iterator, 256> cur_iters; // Iterators for PUCT selection
        std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer; // Buffer for reusing visit arrays
        std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform; // Visits planned per child edge
        std::vector<int> vtp_last_filled; // Tracks last filled index in visits_to_perform
        std::vector<int> current_path; // Path indices for selection state
        std::vector<lczero::Move> moves_to_path; // Moves corresponding to current_path
        // TaskWorkspace constructor needed? Initialize members if necessary.
        // TaskWorkspace() : history_(???) {} // Need initial history if used here
    };

     struct PickingTask {
        Node* node;
        uint16_t base_depth;
        std::vector<lczero::Move> moves;
        int visits; // The number of visits to perform starting from this node

        PickingTask(Node* n, uint16_t depth, const std::vector<lczero::Move>& m, int v)
            : node(n), base_depth(depth), moves(m), visits(v) {}
     };

};


}  // namespace classic
}  // namespace lczero
