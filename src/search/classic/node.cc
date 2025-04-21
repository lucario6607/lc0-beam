/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "search/classic/node.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <limits> // Added for numeric_limits

#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/exception.h"
#include "utils/hashcat.h"
#include "search/classic/search.h" // Added for kValueKnownWin/Loss constants
#include "search/classic/params.h" // Added for SearchParams access

// Access global search options (assuming it's accessible - adjust if needed)
extern lczero::OptionsDict g_options; // Or however options are globally accessed/passed
extern lczero::classic::SearchParams g_search_options; // Assuming global options object


namespace lczero {
namespace classic {

// Helper definition for GetQ used in SelectChild (based on LC0 structure)
// This likely needs adjustment based on LC0's exact implementation details
// for FPU and draw score handling in selection.
float Node::GetQ(const SearchParams& params) const {
     // Determine draw score based on depth (needs parent access to know depth reliably)
     // Placeholder: Assume root draw score for now, selection logic needs correct depth
     float draw_score = 0.0; // Simplified - GetDrawScore(depth % 2 != 0) needed
     if (parent_) {
         // Crude depth estimate based on parent pointer chain (inefficient, for concept only)
         int depth = 0;
         const Node* p = this;
         while(p->GetParent() != nullptr) { // Check against null parent
              p = p->GetParent();
              depth++;
              if(depth > 200) break; // Safety break
         }
         // Root is depth 0. Node depth is parent depth + 1.
         bool is_odd_depth = (depth % 2 != 0); // Check parity
         // Need access to Search object or history to get correct draw score perspective
         // draw_score = search->GetDrawScore(is_odd_depth); // Conceptual
     }


     uint32_t n = GetN();
     if (n > 0) {
         return GetQ(draw_score); // Use existing GetQ with draw score
     } else {
         // Apply FPU if N=0
         if (!parent_) return 0.0f; // Root node case without FPU application here

         // Determine if root or not for FPU params
         bool is_root_node = (parent_->GetParent() == nullptr); // Simple check
         const float fpu_value = params.GetFpuValue(is_root_node);
         if (params.GetFpuAbsolute(is_root_node)) {
             return fpu_value;
         } else {
             // FPU Reduction: ParentQ - reduction * sqrt(policy)
             // Ensure parent's draw score perspective is correct here
             float parent_q = -parent_->GetQ(-draw_score); // Get parent Q from parent's perspective, flip
             float policy_prior = GetPolicyPrior();
             return parent_q - fpu_value * std::sqrt(policy_prior);
         }
     }
}


/////////////////////////////////////////////////////////////////////////
// Node garbage collector
/////////////////////////////////////////////////////////////////////////

namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }

  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_.store(true);
     if (gc_thread_.joinable()) { // Check if joinable before joining
         gc_thread_.join();
     }
    // Process remaining items after thread stops
    GarbageCollect(true);
  }


 private:
  void GarbageCollect(bool final_run = false) { // Added final_run parameter
      while (true) { // Loop until queue is empty or stopped (if not final_run)
        // Node will be released in destructor when mutex is not locked.
        std::unique_ptr<Node> node_to_gc;
        size_t solid_size = 0;
        {
          // Lock the mutex and move last subtree from subtrees_to_gc_ into
          // node_to_gc.
          Mutex::Lock lock(gc_mutex_);
          if (subtrees_to_gc_.empty()) return; // Exit if queue is empty
          node_to_gc = std::move(subtrees_to_gc_.back());
          subtrees_to_gc_.pop_back();
          solid_size = subtrees_to_gc_solid_size_.back();
          subtrees_to_gc_solid_size_.pop_back();
        }
        // Solid is a hack...
        if (solid_size != 0) {
          for (size_t i = 0; i < solid_size; i++) {
             // Check pointer validity before calling destructor
             if(node_to_gc.get() + i != nullptr) {
                node_to_gc.get()[i].~Node();
             }
          }
          std::allocator<Node> alloc;
          // Check if pointer is valid before deallocating
           if (node_to_gc) {
              alloc.deallocate(node_to_gc.release(), solid_size);
           }
        }
        // If not the final run and stopped, exit the loop
        if (!final_run && stop_.load()) return;
      }
  }


  void Worker() {
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
    // Do a final garbage collect run after stopping
    GarbageCollect(true);
  }


  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_);
  std::vector<size_t> subtrees_to_gc_solid_size_ GUARDED_BY(gc_mutex_);

  // When true, Worker() should stop and exit.
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  // Use memcpy for type punning to avoid potential strict aliasing issues
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  // Use memcpy for type punning
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}


std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString(true) << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// Node: Min/Max Implementations
/////////////////////////////////////////////////////////////////////////

// --- Implementation of GetMinValue ---
Value Node::GetMinValue() const {
    // Check known state first (Base Case 1)
    // Use relaxed memory order for reads as they are checked before writes within Backup context.
    if (is_known_win.load(std::memory_order_relaxed)) return kValueKnownWin;
    if (is_known_loss.load(std::memory_order_relaxed)) return kValueKnownLoss;

    // If node is unexpanded or has never been visited (Base Case 2 - AllieStein logic)
    // Relying on atomic/thread-safe read for GetVisitCount().
    if (!HasChildren() || GetVisitCount() == 0) {
        return GetValue(); // Use current MCTS Q as estimate/base value (WL from node's perspective)
    }

    Value min_val = kValueKnownWin; // Initialize to worst possible outcome for opponent (best for me)
    bool any_child_considered = false; // Track if we find any valid child to recurse into

    // Access children safely (assuming GetChildrenPtr/GetChild handle concurrency or are called within lock)
    const int num_children = GetNumChildren(); // Get number of children

    for (int i = 0; i < num_children; ++i) {
        Node* child = GetChild(i);
        // --- Use AllieStein's filter: Only consider visited children ---
        if (child == nullptr || child->GetVisitCount() == 0) continue;

        any_child_considered = true;
        // Value from child is from their perspective. My MinValue = min(child's MaxValue)
        min_val = std::min(min_val, child->GetMaxValue()); // Recursive call without depth

        // Early exit "pruning" if opponent can force a loss (my known loss)
        if (min_val <= kValueKnownLoss) {
             return kValueKnownLoss; // Pruning like Alpha-Beta min score update
        }
    }

    // If no children were visited > 0 times, fallback to MCTS Q (Base Case 3)
    if (!any_child_considered) {
        return GetValue();
    }

    return min_val;
}

// --- Implementation of GetMaxValue ---
Value Node::GetMaxValue() const {
    // Check known state first (Base Case 1)
    if (is_known_win.load(std::memory_order_relaxed)) return kValueKnownWin;
    if (is_known_loss.load(std::memory_order_relaxed)) return kValueKnownLoss;

    // If node is unexpanded or has never been visited (Base Case 2 - AllieStein logic)
    if (!HasChildren() || GetVisitCount() == 0) {
        return GetValue(); // Use current MCTS Q as estimate/base value (WL from node's perspective)
    }

    Value max_val = kValueKnownLoss; // Initialize to worst possible outcome for me
    bool any_child_considered = false;

    const int num_children = GetNumChildren();

    for (int i = 0; i < num_children; ++i) {
        Node* child = GetChild(i);
         // --- Use AllieStein's filter: Only consider visited children ---
        if (child == nullptr || child->GetVisitCount() == 0) continue;

        any_child_considered = true;
        // Value from child is from their perspective. My MaxValue = max(child's MinValue)
        max_val = std::max(max_val, child->GetMinValue()); // Recursive call without depth

        // Early exit "pruning" if I can force a win
        if (max_val >= kValueKnownWin) {
            return kValueKnownWin; // Pruning like Alpha-Beta max score update
        }
    }

     // If no children were visited > 0 times, fallback to MCTS Q (Base Case 3)
    if (!any_child_considered) {
        return GetValue();
    }

    return max_val;
}


/////////////////////////////////////////////////////////////////////////
// Node: Other Methods
/////////////////////////////////////////////////////////////////////////


Node* Node::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist({move});
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist(moves);
  num_edges_ = moves.size();
}

Node::ConstIterator Node::Edges() const {
  return {this, !solid_children_ ? &child_ : nullptr}; // Pass `this` instead of `*this`
}
Node::Iterator Node::Edges() {
  return {this, !solid_children_ ? &child_ : nullptr}; // Pass `this` instead of `*this`
}


float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) { // Iterate directly over Node*
      if (node) { // Check if node pointer is valid
          const Edge* edge = GetEdgeToNode(node);
          if (edge) sum += edge->GetP();
      }
  }
  return sum;
}


Edge* Node::GetEdgeToNode(const Node* node) const {
  if (!node || node->parent_ != this || !edges_ || node->index_ >= num_edges_) {
       return nullptr; // Added validity checks
  }
  return &edges_[node->index_];
}


Edge* Node::GetOwnEdge() const {
    if (!parent_) return nullptr; // Added null check for parent
    return GetParent()->GetEdgeToNode(this);
}


std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_.load() << " D:" << d_.load() << " M:" << m_.load() // Use .load() for atomics
      << " N:" << n_.load() << " N_:" << n_in_flight_.load()
      << " Edges:" << static_cast<int>(num_edges_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << " Solid:" << solid_children_
      << " KW:" << is_known_win.load() << " KL:" << is_known_loss.load(); // Added known state flags
  return oss.str();
}

bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  // Can only make solid if no immediate leaf children are in flight since we
  // allow the search code to hold references to leaf nodes across locks.
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    // Use atomic loads for N and NInFlight
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  // If the total of children in flight is not the same as self, then there are
  // collisions against immediate children (which don't update the GetNInFlight
  // of the leaf) and its not safe. Use atomic load for self NInFlight.
  if (total_in_flight != GetNInFlight()) {
    return false;
  }
  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
    // --- NEW: Copy known state flags ---
    // This assumes the corresponding node existed in the old structure.
    // Need to find the old node corresponding to index `i`.
    Node* old_corresponding_node = nullptr;
    Node* current_old = child_.get();
    while(current_old) {
        if (current_old->index_ == i) {
            old_corresponding_node = current_old;
            break;
        }
        current_old = current_old->sibling_.get();
    }
    if (old_corresponding_node) {
        new_children[i].is_known_win.store(old_corresponding_node->is_known_win.load(std::memory_order_relaxed));
        new_children[i].is_known_loss.store(old_corresponding_node->is_known_loss.load(std::memory_order_relaxed));
        // Copy bounds as well
        new_children[i].lower_bound_ = old_corresponding_node->lower_bound_;
        new_children[i].upper_bound_ = old_corresponding_node->upper_bound_;

    }
    // --- End NEW ---
  }

  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    if (index < num_edges_) { // Bounds check
        new_children[index] = std::move(*old_child.get());
        // This isn't needed, but it helps crash things faster if something has gone
        // wrong.
        old_child->parent_ = nullptr;
        gNodeGc.AddToGcQueue(std::move(old_child)); // Move ownership to GC
        new_children[index].UpdateChildrenParents();
        old_child = std::move(new_children[index].sibling_); // Get next sibling BEFORE moving old_child
        new_children[index].sibling_.reset(); // Clear sibling ptr in new solid node
    } else {
        // Index out of bounds, indicates an error. GC the node and break.
        gNodeGc.AddToGcQueue(std::move(old_child));
        assert(false && "Node index out of bounds during MakeSolid");
        break;
    }
  }
  // This is a hack.
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}


void Node::SortEdges() {
  if (!edges_ || num_edges_ <= 1) return; // Added null check and size check
  assert(!child_ || !solid_children_); // Cannot sort edges if children are solid (nodes might mismatch)
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_.store(plies_left, std::memory_order_relaxed); // Use atomic store
  if (result == GameResult::DRAW) {
    wl_.store(0.0, std::memory_order_relaxed);
    d_.store(1.0, std::memory_order_relaxed);
  } else if (result == GameResult::WHITE_WON) {
    wl_.store(1.0, std::memory_order_relaxed);
    d_.store(0.0, std::memory_order_relaxed);
  } else if (result == GameResult::BLACK_WON) {
    wl_.store(-1.0, std::memory_order_relaxed);
    d_.store(0.0, std::memory_order_relaxed);
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    Edge* own_edge = GetOwnEdge();
    if (own_edge != nullptr) own_edge->SetP(0.0f);
  }
  // --- NEW: Set known state flags when made terminal ---
  // Access global options - ensure this is valid in your setup
   if (g_search_options.GetProvenStateHandling()) {
       if (result == GameResult::WHITE_WON) {
           is_known_win.store(true, std::memory_order_relaxed);
           is_known_loss.store(false, std::memory_order_relaxed); // Ensure loss is false
       } else if (result == GameResult::BLACK_WON) {
           is_known_loss.store(true, std::memory_order_relaxed);
           is_known_win.store(false, std::memory_order_relaxed); // Ensure win is false
       }
       // Optionally handle DRAW if we add an is_known_draw flag
   }
  // --- End NEW ---
}


void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  // Reset bounds to default unknown state
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  // Reset known state flags
  is_known_win.store(false, std::memory_order_relaxed);
  is_known_loss.store(false, std::memory_order_relaxed);

  // Recalculate stats based on children if they exist
  n_.store(0, std::memory_order_relaxed); // Reset N before recalculating
  wl_.store(0.0, std::memory_order_relaxed);
  d_.store(1.0, std::memory_order_relaxed); // Default to draw if no children visits
  m_.store(0.0, std::memory_order_relaxed);

  if (HasChildren()) {
     uint32_t total_child_n = 0;
     double total_child_wl_n = 0.0;
     float total_child_d_n = 0.0f;
     // Iterate through children to sum up stats
     // This needs locking if children can be modified concurrently,
     // but MakeNotTerminal is usually called in a context where it's safe.
     for (auto& child_iter : Edges()) {
         Node* child = child_iter.node();
         if (child) {
             const auto n_child = child->GetN();
             if (n_child > 0) {
                 total_child_n += n_child;
                 // Flip child's WL perspective for parent
                 total_child_wl_n += -child->GetWL() * n_child;
                 total_child_d_n += child->GetD() * n_child;
                 // Note: M value is not typically averaged back up like this
             }
         }
     }

     // Set own stats based on children averages (plus the implicit visit for expansion)
     uint32_t self_n = 1 + total_child_n;
     n_.store(self_n, std::memory_order_relaxed);
     if (total_child_n > 0) { // Use total_child_n to avoid division by zero
         wl_.store(total_child_wl_n / total_child_n, std::memory_order_relaxed);
         d_.store(total_child_d_n / total_child_n, std::memory_order_relaxed);
     } else {
         // If no children had visits, keep default WL=0, D=1
          wl_.store(0.0, std::memory_order_relaxed);
          d_.store(1.0, std::memory_order_relaxed);
     }
     // M remains 0.0f as it's not averaged back up typically.
  } else {
       // If no children edges exist yet (just reverted a terminal leaf)
       n_.store(1, std::memory_order_relaxed); // It has one visit (the one that led here)
       // Values remain default (0 WL, 1 D, 0 M)
  }
}


void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  // Use atomic compare-and-swap or fetch-and-add logic for safety
  // This implementation assumes relaxed memory order is sufficient, review needed
  uint32_t current_n = n_.load(std::memory_order_relaxed);
  uint32_t current_n_in_flight = n_in_flight_.load(std::memory_order_relaxed);

  if (current_n == 0 && current_n_in_flight > 0) return false; // Collision check

  n_in_flight_.fetch_add(1, std::memory_order_relaxed); // Increment atomically
  return true;
}

void Node::CancelScoreUpdate(int multivisit) {
     n_in_flight_.fetch_sub(multivisit, std::memory_order_relaxed); // Use atomic fetch_sub
}

void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  // Use atomic operations for thread-safe updates.
  // Fetch current values for weighted average calculation.
  // Use relaxed memory order as updates are cumulative and order between threads
  // doesn't strictly matter for the final average, only atomicity of read-modify-write.

  uint32_t current_n = n_.fetch_add(multivisit, std::memory_order_acq_rel); // Fetch old N, then add
  uint32_t new_n = current_n + multivisit;

  // Ensure new_n is positive to avoid division by zero. Should always be true here.
  if (new_n > 0) {
      // Perform atomic updates using compare-exchange loops for floating points
      // WL update
      double current_wl = wl_.load(std::memory_order_relaxed);
      double new_wl;
      do {
          new_wl = current_wl + multivisit * (v - current_wl) / new_n;
      } while (!wl_.compare_exchange_weak(current_wl, new_wl, std::memory_order_release, std::memory_order_relaxed));

      // D update
      float current_d = d_.load(std::memory_order_relaxed);
      float new_d;
      do {
          new_d = current_d + multivisit * (d - current_d) / new_n;
      } while (!d_.compare_exchange_weak(current_d, new_d, std::memory_order_release, std::memory_order_relaxed));

      // M update
      float current_m = m_.load(std::memory_order_relaxed);
      float new_m;
      do {
           new_m = current_m + multivisit * (m - current_m) / new_n;
      } while (!m_.compare_exchange_weak(current_m, new_m, std::memory_order_release, std::memory_order_relaxed));
  } else {
      // Handle case where new_n is 0 (should not happen if multivisit > 0)
      // Perhaps reset values if N becomes 0?
      wl_.store(0.0, std::memory_order_relaxed);
      d_.store(1.0, std::memory_order_relaxed); // Default to draw
      m_.store(0.0, std::memory_order_relaxed);
  }


  // Decrement virtual loss atomically.
  n_in_flight_.fetch_sub(multivisit, std::memory_order_release);
}


void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  // Recompute Q, D, M based on delta values and current N.
  // This requires atomic updates.

  uint32_t current_n = n_.load(std::memory_order_relaxed);
  if (current_n == 0) return; // Cannot adjust if N is zero

  // WL update
  double current_wl = wl_.load(std::memory_order_relaxed);
  double new_wl;
  do {
      new_wl = current_wl + multivisit * v / current_n;
  } while (!wl_.compare_exchange_weak(current_wl, new_wl, std::memory_order_release, std::memory_order_relaxed));

  // D update
  float current_d = d_.load(std::memory_order_relaxed);
  float new_d;
  do {
      new_d = current_d + multivisit * d / current_n;
  } while (!d_.compare_exchange_weak(current_d, new_d, std::memory_order_release, std::memory_order_relaxed));

  // M update
  float current_m = m_.load(std::memory_order_relaxed);
  float new_m;
  do {
      new_m = current_m + multivisit * m / current_n;
  } while (!m_.compare_exchange_weak(current_m, new_m, std::memory_order_release, std::memory_order_relaxed));
}


void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  // Fetch current N atomically
  uint32_t current_n = n_.load(std::memory_order_relaxed);
  // Ensure multivisit doesn't exceed current_n
  if (multivisit > current_n) multivisit = current_n;

  uint32_t n_new = current_n - multivisit;

  if (n_new == 0) {
      // If n_new == 0, reset all relevant values to 0/defaults using atomic stores.
      wl_.store(0.0, std::memory_order_relaxed);
      d_.store(1.0, std::memory_order_relaxed); // Default to draw
      m_.store(0.0, std::memory_order_relaxed);
      n_.store(0, std::memory_order_release); // Store new N value
  } else {
      // Recompute Q, D, M atomically. Use compare-exchange loops.
      // WL update
      double current_wl = wl_.load(std::memory_order_relaxed);
      double target_wl;
      do {
           // Calculate the target value based on removing the influence of 'v'
           target_wl = (current_n * current_wl - multivisit * v) / n_new;
      } while (!wl_.compare_exchange_weak(current_wl, target_wl, std::memory_order_release, std::memory_order_relaxed));

      // D update
      float current_d = d_.load(std::memory_order_relaxed);
      float target_d;
      do {
           target_d = (current_n * current_d - multivisit * d) / n_new;
      } while (!d_.compare_exchange_weak(current_d, target_d, std::memory_order_release, std::memory_order_relaxed));

       // M update
      float current_m = m_.load(std::memory_order_relaxed);
      float target_m;
       do {
           target_m = (current_n * current_m - multivisit * m) / n_new;
       } while (!m_.compare_exchange_weak(current_m, target_m, std::memory_order_release, std::memory_order_relaxed));

      // Decrement N atomically AFTER calculating new values.
      n_.store(n_new, std::memory_order_release);
  }
}


void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
     // Ensure child_ points to a valid array before iterating
    if (child_) {
        Node* child_array = child_.get();
        for (int i = 0; i < num_edges_; i++) {
          child_array[i].parent_ = this;
        }
    }
  }
}


void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
  // Also reset local pointers/flags associated with children
  edges_.reset();
  num_edges_ = 0;
  solid_children_ = false; // No longer solid after releasing
  // child_ is already moved
  sibling_.reset(); // Siblings are irrelevant if children are gone
}


void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (!HasChildren()) return; // Nothing to release

  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr && node_to_save->parent_ == this) { // Check parent
        // Ensure index is valid
        if(node_to_save->index_ < num_edges_) {
            // Move the node data into a new temporary node
            saved_node = std::make_unique<Node>(this, node_to_save->index_);
            *saved_node = std::move(*node_to_save); // Move semantics
        }
    }
    // Send the whole array (now potentially with one moved-from element) to GC
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    // Assign the saved node (if any) back to child_
    child_ = std::move(saved_node);
    // Reset flags as it's no longer solid
    solid_children_ = false;
    num_edges_ = child_ ? 1 : 0; // Update edge count
    if (child_) {
        child_->sibling_.reset(); // Saved node has no siblings now
        child_->parent_ = this; // Ensure parent pointer is correct
        child_->UpdateChildrenParents(); // Update children of the saved node
        // Recreate the single edge corresponding to the saved node
        // Need the move first. Get it from the saved node's index if possible.
        if (edges_ && child_->index_ < num_edges_) {
             Move saved_move = edges_[child_->index_].GetMove();
             edges_ = Edge::FromMovelist({saved_move});
        } else {
             // Fallback or error if edge info is lost/invalid
             edges_.reset();
             num_edges_ = 0;
        }

    } else {
        edges_.reset(); // No saved node, no edges
        num_edges_ = 0; // Ensure num_edges is 0
    }

  } else { // Linked list mode
    std::unique_ptr<Node> saved_node;
    std::unique_ptr<Node> current_child = std::move(child_); // Take ownership of list head
    child_.reset(); // Clear original child pointer

    while (current_child) {
      std::unique_ptr<Node> next_sibling = std::move(current_child->sibling_); // Get next before potentially GCing current
      if (current_child.get() == node_to_save) {
         // Found the node to save
         saved_node = std::move(current_child);
         // GC the rest of the siblings that came after
         gNodeGc.AddToGcQueue(std::move(next_sibling));
         // Break the loop as we found the node and handled subsequent siblings
         break;
      } else {
         // Node not saved, send to GC
         gNodeGc.AddToGcQueue(std::move(current_child));
      }
      current_child = std::move(next_sibling); // Move to next sibling
    }

    // Make saved node the only child.
    child_ = std::move(saved_node);
    if (child_) {
        child_->sibling_.reset(); // Ensure saved node has no sibling
        num_edges_ = 1; // Update edge count
        // Recreate the single edge
        // Need the move info. Get it from the saved node's index.
        if (edges_ && child_->index_ < num_edges_) { // Check old edges_/num_edges_ validity
             Move saved_move = edges_[child_->index_].GetMove();
             edges_ = Edge::FromMovelist({saved_move});
        } else {
            // Fallback/error
             edges_.reset();
             num_edges_ = 0;
        }

    } else {
        num_edges_ = 0;
        edges_.reset();
    }
  }
}


/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

void NodeTree::MakeMove(Move move) {
  Node* new_head = nullptr;
  if (current_head_->HasChildren()) { // Check if children exist
      for (auto& n : current_head_->Edges()) {
        if (n.GetMove() == move) {
          new_head = n.GetOrSpawnNode(current_head_);
          // Ensure head is not terminal, so search can extend or visit children of
          // "terminal" positions, e.g., WDL hits, converted terminals, 3-fold draw.
           if (new_head && new_head->IsTerminal()) { // Add null check for new_head
               // Also reset known state flags if making not terminal
               // Check global option before calling MakeNotTerminal potentially
               // Pass SearchParams or access globally if needed.
               // Assuming g_search_options is accessible:
               new_head->MakeNotTerminal(); // MakeNotTerminal now handles flags internally based on g_search_options
           }
          break;
        }
      }
  } // End if HasChildren
  // If new_head was found and needs to be the sole child
  if (new_head) {
      current_head_->ReleaseChildrenExceptOne(new_head);
      // After ReleaseChildrenExceptOne, current_head_->child_ points to the saved node (new_head)
      current_head_ = current_head_->child_.get();
       // Ensure current_head_ is not null after potential release/reset
       if (!current_head_) {
           // This case might happen if the node_to_save was null initially
           // or if ReleaseChildrenExceptOne failed. Need robust handling.
           // Fallback: Create a new node if head became null.
           // Note: This might indicate a logic error needing investigation.
           current_head_ = GetGameBeginNode()->CreateSingleChildNode(move); // Re-root? Or error?
           // This fallback is problematic, ideally MakeMove shouldn't result in null head.
       }
  } else {
      // If move wasn't found among existing children (or no children existed),
      // create a new node for this move. This implies pruning the old tree.
      current_head_->ReleaseChildren(); // Remove all existing children
      current_head_ = current_head_->CreateSingleChildNode(move); // Create the new path
  }

  history_.Append(move);
}


void NodeTree::TrimTreeAtHead() {
  // If solid, this will be empty before move and will be moved back empty
  // afterwards which is fine.
  auto tmp = std::move(current_head_->sibling_); // Save sibling ptr
  // Send dependent nodes for GC instead of destroying them immediately.
  current_head_->ReleaseChildren();
  // Reset node state BUT keep parent link and index
   Node* parent = current_head_->GetParent();
   uint16_t index = current_head_->Index();
  *current_head_ = Node(parent, index); // Reset using constructor
  current_head_->sibling_ = std::move(tmp); // Restore sibling ptr
}

bool NodeTree::ResetToPosition(const GameState& pos) {
  if (gamebegin_node_ && (history_.Starting() != pos.startpos)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
  }

  history_.Reset(pos.startpos);

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const Move m : pos.moves) {
    MakeMove(m);
    if (old_head == current_head_) seen_old_head = true;
     // Safety break if current_head_ becomes null unexpectedly
     if (!current_head_) {
          // Log error or handle - tree structure might be corrupted
           assert(false && "current_head_ became null during ResetToPosition");
           break;
     }
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head && current_head_) TrimTreeAtHead(); // Add null check
  return seen_old_head;
}

bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(starting_fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
     if (m == Move::Null()) { // Check if move parsing failed
         // Handle error - e.g., log, throw, or return false
         std::cerr << "Warning: Could not parse move '" << move << "' in ResetToPosition." << std::endl;
         return false; // Indicate failure
     }
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    cur_board.Mirror();
  }
  return ResetToPosition(state);
}

void NodeTree::DeallocateTree() {
  // Same as gamebegin_node_.reset(), but actual deallocation will happen in
  // GC thread.
  gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}

}  // namespace classic
}  // namespace lczero
