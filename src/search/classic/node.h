/*
  This file is part of Leela Chess Zero.
  Copyright (C) 20 incorporating the necessary includes, type fixes, atomic members, Min/Max declarations, helper methods, and reordering fixes.

```c++
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

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>       // Added for std::atomic
#include <vector>       // Added for std::vector
#include <limits>       // Added for numeric_limits
#include <utility>      // Added for std::pair

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h" // Include for PositionHash (if defined here)
#include "chess/chess.h"    // Include for Value, GameResult, kValueMate etc.
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"

namespace lczero {
namespace classic {

18 The LCZero Authors

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

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic> // Added for std::atomic
#include <vector> // Added for std::vector
#include <limits> // Added for numeric_limits

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h" // Include for PositionHash (if defined here or needed indirectly)
#include "chess/chess.h"     // Include for Value, GameResult, kValueMate, Move etc.
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"


namespace lczero {
namespace classic {

// Forward declaration
class SearchParams; // Needs to be defined where SearchParams is used if needed

class Node; // Forward declare Node

class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

class EdgeAndNode; // Forward declare EdgeAndNode

// Template declarations for iterators
template <bool is_const> class Edge_Iterator;
template <bool is_const> class VisitedNode_Iterator;


class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(// Forward declaration
class SearchParams; // Needed if SearchParams used in methods like GetQ

// Children of a node... (Comment remains the same) ...

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

 // Make members private and provide accessors if needed, or keep protected/public based on usage
 // Keeping them private for encapsulation example:
 private:
  Move move_;
  uint16_t p_ = 0;
  friend class Node; // Node needs access to Edge members
  friend class Edge_Iterator<true>; // Iterators might need access
  friend class Edge_Iterator<false>;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {
      // Optional TT loading logic would go here if constructing from TTEntry
  }

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  Node* CreateSingleChildNode(Move m);
  void CreateEdges(const MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); }
  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0;
  }
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const {
      return GetParent() ? GetParent()->GetNStarted() : 0;
   }

  float GetQ(float draw_score) const {
       uint32_t n = GetN();
       if (n == 0) return 0.0f; // Return 0 if no visits, FPU handled elsewhere
       // Use atomic loads
       double current_wl = wl_.load(std::memory_order_relaxed);
       float current_d = d_.load(std::memory_order_relaxed);
       return static_cast<float>(current_wl + draw_score * current_d);
   }
  // Returns node WL value (perspective of player to move)
  Value GetValue() const {
      uint32_t n = GetN();
      if (n == 0) return kValueZero; // Return 0 for unvisited
      return static_cast<Value>(wl_.load(std::memory_order_relaxed));
  }
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  float GetM() const { return m_.load(std::memory_order_relaxed); }

  // Get Policy Prior from the parent edge leading to this node
  float GetPolicyfalse) {}

  // Default move constructor/assignment is fine
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  // Delete copy constructor/assignment
  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;

  // Allocates a new edge and a new node. The node must have no edges before that.
  Node* CreateSingleChildNode(Move m);

  // Creates edges from a movelist. There must be no edges before that.
  void CreateEdges(const MoveList& moves);

  // Gets parent node.
  Node* GetParent() const { return parent_; }

  // Returns whether a node has children edges allocated.
  bool HasChildren() const { return static_cast<bool>(edges_); }

  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;

  // Visit count accessors (using atomic relaxed loads)
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  uint32_t GetNInFlight() const { return n_in_flight_.load(std::memory_order_relaxed); }
  uint32_t GetVisitCount() const { return GetN(); } // Helper alias

  // Get combined visits (N + N_in_flight) atomically
  int GetNStarted() const { return n_.load(std::memory_order_relaxed) + n_in_flight_.load(std::memory_order_relaxed); }

  // Get number of visits among children
  uint32_t GetChildrenVisits() const {
      uint32_t n = GetN();
      return n > 0 ? n - 1 : 0;
  }

  // Effective visits for PUCT calculations (thread-safe)
  int GetEffectiveVisits() const { return GetNStarted(); }
  int GetEffectiveParentVisits() const; // Needs parent access, implement in .cc

  // Value accessors (using atomic relaxed loads)
  // Returns node eval Q = WL + draw_score * D
  float GetQ(float draw_score) const {
       uint32_t n = GetN();
       if (n == 0) return 0.0f; // FPU handled in selection logic
       double current_wl = wl_.load(std::memory_order_relaxed);
       float current_d = d_.load(std::memory_order_relaxed);
       return static_cast<float>(current_wl + draw_score * current_d);
   }
  // Returns node eval based on WL only (perspective of player to move)
  Value GetValue() const { // Returns Value (double)
      uint32_t n = GetN();
      Prior() const;

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(Value v, float d, float m, int multivisit);
  void AdjustForTerminal(Value v, float d, float m, int multivisit);
  void RevertTerminalVisits(Value v, float d, float m, int multivisit);
  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multivisit, std::memory_order_relaxed); }
  void UpdateMaxDepth(int depth); // Implementation might be in .cc
  bool UpdateFullDepth(uint16_t* depth); // Implementation might be in .cc
  ConstIterator Edges() const;
  Iterator Edges();
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();
  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();
  uint16_t Index() const { return index_; }

   // --- NEW MEMBERS for Proven State ---
   std::atomic<bool> is_known_win{false};
   std::atomic<bool> is_known_loss{false};

   // --- NEW METHODS for Proven State ---
   Value GetMinValue() const; // Use Value type
   Value GetMaxValue() const; // Use Value type

   // --- Helper Methods ---
   const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_; }
   Node* GetChild(int index) const;
   int GetNumChildren() const { return num_edges_; }


  ~Node() {
    if (solid_children_ && child_) {
      for (int i = 0; i < num_edges_; i++) {
         if (child_.get() + i != nullptr) {
            child_.get()[i].~Node();
         }
      }
      std::allocator<Node> alloc;
      if (child_) {
          alloc.deallocate(child_.release(), num_edges_);
      }
    }
  }

 private:
  if (n == 0) return 0.0; // Return 0.0 for unvisited
      return wl_.load(std::memory_order_relaxed);
  }
  // Returns node WL eval (W-L) as float
  float GetWL() const { return static_cast<float>(wl_.load(std::memory_order_relaxed)); }
  // Returns node Draw probability as float
  float GetD() const { return d_.load(std::memory_order_relaxed); }
  // Returns node Moves Left estimate as float
  float GetM() const { return m_.load(std::memory_order_relaxed); }

  // Get Policy Prior for the edge leading to this node (from parent)
  float GetPolicyPrior() const;

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }

  // Node bounds (best/worst possible outcome)
  typedef std::pair<GameResult,void UpdateChildrenParents();

  // Member variable order (largest to smallest, atomics grouped)
  // 8 byte fields.
  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  std::atomic<double> wl_{0.0}; // Using atomic double

  // 4 byte fields.
  std::atomic<float> d_{0.0};
  std::atomic<float> m_{0.0};
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};

  // 2 byte fields.
 GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }

  // Number of legal moves/edges from this node
  uint8_t GetNumEdges() const { return num_edges_; }

  // Copies policy priors for the first max_needed edges into output buffer.
  void CopyPolicy(int max_needed, float* output) const;

  // Makes the node terminal and sets its score and  uint16_t index_;

  // 1 byte fields.
  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  bool solid_children_ type.
  void MakeTerminal(GameResult result, float plies_left = 0.0f, : 1;
  // is_known_win/loss flags already declared above

  friend class NodeTree;
                    Terminal type = Terminal::EndOfGame);
  // Makes the node non-terminal and recalculates its stats
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class based on children.
  void MakeNotTerminal();

  // Sets the node's bounds explicitly.
  void Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator< SetBounds(GameResult lower, GameResult upper);

  // Tries to start processing this node (for virtualfalse>;
  friend class SearchWorker; // Allow SearchWorker access
};

// Re-check assertion after compiling with atomics
// static_assert(sizeof(Node) == 64, "Unexpected size of Node"); loss). Returns true if successful.
  bool TryStartScoreUpdate();
  // Cancels a previous score update (rem

class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edgeoves virtual loss).
  void CancelScoreUpdate(int multivisit);

  // Finalizes MCTS backup:* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge updates N, N_in_flight, and averages WL, D, M.
  void FinalizeScoreUpdate_ = nullptr; node_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
   (Value v, float d, float m, int multivisit); // Use Value type
  // Adjusts WLbool operator==(const EdgeAndNode& other) const {
       return edge_ == other.edge_ && node_ == other.node_;
   }
   bool operator!=(const EdgeAndNode& other) const {, D, M based on a delta from a child becoming terminal.
  void AdjustForTerminal(Value v_delta, float return !(*this == other); }
   bool HasNode() const { return node_ != nullptr; }
 d_delta, float m_delta, int multivisit); // Use Value type
  // Reverts visit statistics   Edge* edge() const { return edge_; }
   Node* node() const { return node_; }

 when a terminal node is made non-terminal.
  void RevertTerminalVisits(Value v, float d, float m   float GetQ(float default_q, float draw_score) const {
     return (node_ && node_->Get, int multivisit); // Use Value type

  // Increments N_in_flight (adds virtual loss).
N() > 0) ? node_->GetQ(draw_score) : default_q;
   }  void IncrementNInFlight(int multivisit) { n_in_flight_.fetch_add(multiv
   float GetWL(float default_wl) const {
     return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
   }
   float GetDisit, std::memory_order_relaxed); }

  // These are likely defined elsewhere or not used directly
  // void(float default_d) const {
     return (node_ && node_->GetN() > 0) UpdateMaxDepth(int depth);
  // bool UpdateFullDepth(uint16_t* depth);

  // Returns ? node_->GetD() : default_d;
   }
   float GetM(float default_m range for iterating over edges.
  ConstIterator Edges() const;
  Iterator Edges();

  // Returns) const {
     return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
   }
   uint32_t GetN() const { return node_ range for iterating over child nodes with N > 0.
  VisitedNode_Iterator<true> VisitedNodes ? node_->GetN() : 0; }
   int GetNStarted() const { return node_ ?() const;
  VisitedNode_Iterator<false> VisitedNodes();

  // Deletes all children and node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const edges.
  void ReleaseChildren();

  // Deletes all children except the specified one.
  void Release { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal() constChildrenExceptOne(Node* node_to_save);

  // For a child node, returns the corresponding edge { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const { return from the parent.
  Edge* GetEdgeToNode(const Node* node) const;

  // Returns the node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const {
 edge in the parent node that leads to this node.
  Edge* GetOwnEdge() const;

  // Debug information     return node_ ? node_->GetBounds()
                  : Node::Bounds{GameResult::BLACK_WON, about the node.
  std::string DebugString() const;

  // Optimizes child node storage for cache GameResult::WHITE_WON};
   }
   float GetP() const { return edge_ ? edge_->GetP() : 0.0f; }
   Move GetMove(bool flip = false) const {
     return edge_ ? locality. Returns true if successful.
  bool MakeSolid();

  // Sorts edges based on policy prior (descending). edge_->GetMove(flip) : Move();
   }
   float GetU(float numerator) const {
  void SortEdges();

  // Index of this node in its parent's edge list.
  uint16_t Index() const { return index_; }

   // --- NEW MEMBERS for Proven State ---
   std::atomic<bool>
       return GetP() == 0.0f ? 0.0f : numerator * GetP() / ( is_known_win{false};
   std::atomic<bool> is_known_loss{false};1.0f + static_cast<float>(GetNStarted()));
   }
   std::string DebugString() const

   // --- NEW METHODS for Proven State ---
   // Recursive functions to calculate minimax bounds based on current MCTS values;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};

template
   Value GetMinValue() const; // Use Value type
   Value GetMaxValue() const; // Use Value type

 <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                  // --- NEW Helper Methods ---
   // Provides access to the unique_ptr managing the first child (needed by  std::unique_ptr<Node>*>;
  using NodePtr = std::conditional_t<is_ iterators)
   const std::unique_ptr<Node>* GetChildrenPtr() const { return &child_;const, const Node*, Node*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator() {} }
   // Gets the i-th child node (handles solid/linked list)
   Node* GetChild(int index) const;
   // Gets the number of children (same as GetNumEdges)
   int GetNum

   Edge_Iterator(NodePtr parent_node, Ptr child_ptr) // Use NodePtr
       : EdgeChildren() const { return num_edges_; }


  ~Node() {
    if (solid_children_ && child_)AndNode(parent_node ? parent_node->edges_.get() : nullptr, nullptr), // Handle null parent
          {
      Node* child_array = child_.get(); // Get raw pointer first
      for (int i = parent_node_(parent_node), // Initialize parent first
         node_ptr_(parent_node && !parent_node0; i < num_edges_; i++) {
         // Check pointer validity before calling destructor
         if (child_array->solid_children_ ? &parent_node->child_ : nullptr), // Use node_ptr_ only if not solid + i != nullptr) { // Check array element pointer
            child_array[i].~Node();
         }
       and parent exists
         total_count_(parent_node ? parent_node->num_edges_ : 0)}
      std::allocator<Node> alloc;
      // Check if pointer is valid before deallocating
       // Initialize total_count after parent
         {
     if (!parent_node) { // Handle null parent edge case
         edge_ = nullptr;
         node_ = nullptr;
         current_idx_ = 0if (child_array) { // Use the raw pointer for deallocation check
          // Release ownership from unique_ptr BEFORE de;
         total_count_ = 0;
         node_ptr_ = nullptr;
         return;
     allocating raw pointer
          child_.release();
          alloc.deallocate(child_array, num_edges_);
      }

     if (parent_node_->solid_children_) {
         node_ptr_ = nullptr; // Not used in solid mode for traversal
         edge_ = parent_node_->edges_.get(); // Point to first edge}
    }
    // unique_ptr members (edges_, child_, sibling_) handle non-solid cleanup automatically.
         if (total_count_ > 0 && parent_node_->child_) { // Check if edges/
  }


 private:
  // For each child, ensures that its parent pointer is pointing to this.
  void Updatechild exist
              node_ = parent_node_->child_.get(); // Point to first node in array
         } else {ChildrenParents();

  // Member variable order (largest to smallest, atomics grouped)
  // 8 byte fields.

              edge_ = nullptr; // No edges means end iterator
              node_ = nullptr;
         }
         current_  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::idx_ = 0;
     } else { // Linked list mode
         node_ptr_ = &parent_node_->unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  std::child_;
         edge_ = parent_node_->edges_.get(); // Point to first edge
         current_atomic<double> wl_{0.0}; // Using atomic double for Win-Loss value

  // 4 byteidx_ = 0;
         if (edge_ && node_ptr_) { // Ensure edge exists before calling Actualize
             Actualize();
         } else {
             edge_ = nullptr; // No edges or no child fields.
  std::atomic<float> d_{0.0}; // Draw probability
  std::atomic<float> m_{0.0}; // Moves left estimate
  std::atomic<uint32_t> n_{ pointer -> end iterator
             node_ = nullptr;
         }
     }
   }


  Edge_Iterator<is0}; // Completed visits
  std::atomic<uint32_t> n_in_flight_{0_const> begin() { return *this; }
  Edge_Iterator<is_const> end() {}; // Visits in progress (virtual loss)

  // 2 byte fields.
  uint16_t index_; // return {}; }
  void operator++(); // Implementation in .cc
  Edge_Iterator& operator*() { return Index in parent's edge list

  // 1 byte fields + bitfields
  uint8_t num_ *this; }
  Node* GetOrSpawnNode(Node* parent); // Implementation in .cc

 private:
  edges_ = 0; // Number of edges/children
  Terminal terminal_type_ : 2; // Typevoid Actualize(); // Implementation in .cc

  NodePtr parent_node_ = nullptr; // Store parent node of terminal state, if any
  GameResult lower_bound_ : 2; // Best possible outcome (from pointer (Declared first)
  Ptr node_ptr_ = nullptr; // Pointer to sibling pointer (linked list mode player-to-move perspective)
  GameResult upper_bound_ : 2; // Worst possible outcome
)
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

template <bool is_const>
class VisitedNode_Iterator {
  bool solid_children_ : 1; // Flag for optimized child storage

  // Note: is_known_win public:
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;/loss are declared above with other atomics

  // Friend declarations
  friend class NodeTree;
  friend class Edge_

  VisitedNode_Iterator() {}

   VisitedNode_Iterator(NodePtr parent_node) // Takes parent directly
       Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class: parent_node_(parent_node), // Initialize parent first
         solid_(parent_node ? parent_node->solid_ VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
  friend classchildren_ : false), // Initialize solid after parent
         total_count_(parent_node ? parent_node-> SearchWorker;
};

// Re-check assertion after compiling with atomics. It might be larger now.
// staticnum_edges_ : 0) // Initialize total_count after parent
         {
         // Fix reorder: Initialize node_ptr_ and current_idx_ last
         if (!parent_node) {
             node_ptr_ = nullptr;
             current_idx_ = 0;
             return;
         }
         if (solid_) {_assert(sizeof(Node) == 64, "Unexpected size of Node");

// Contains Edge and Node pair and set of proxy functions to simplify access
// to them.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge),
             node_ptr_ = parent_node_->child_.get(); // Point to start of array
             current_idx_ node_(node) {}
  void Reset() { edge_ = nullptr; node_ = nullptr; }
   = 0;
              // Find first visited node
              while (current_idx_ < total_count_ &&explicit operator bool() const { return edge_ != nullptr; }
   bool operator==(const EdgeAndNode& other) const { (!node_ptr_ || node_ptr_[current_idx_].GetN() == 0)) {

       return edge_ == other.edge_ && node_ == other.node_;
   }
   bool                  current_idx_++;
              }
              if (current_idx_ >= total_count_) {
 operator!=(const EdgeAndNode& other) const { return !(*this == other); }
   bool HasNode() const { return node_ != nullptr; }
   Edge* edge() const { return edge_; }
                   // Set to null to match end() comparison state
                   node_ptr_ = nullptr;
                   current_idx_ = total_count_; // Set index consistently
              }
         } else { // Linked list mode
              node_ptr   Node* node() const { return node_; }

   // Proxy functions for easier access to node/edge properties
   float GetQ(float default_q, float draw_score) const {
     return (node__ = parent_node_->child_.get(); // Start from first child
              // Find first visited node
              while (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
                  node && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q_ptr_ = node_ptr_->sibling_.get();
              }
              // current_idx_ not strictly needed for traversal;
   }
   float GetWL(float default_wl) const {
     return (node_ && in linked list mode
              current_idx_ = node_ptr_ ? node_ptr_->index_ : total node_->GetN() > 0) ? node_->GetWL() : default_wl;
   }
   float GetD(float default_d) const {
     return (node_ && node_->GetN()_count_;
         }
   }

  bool operator==(const VisitedNode_Iterator<is_const>& other > 0) ? node_->GetD() : default_d;
   }
   float GetM() const; // Implementation in .cc
  bool operator!=(const VisitedNode_Iterator<is_const>&float default_m) const {
     return (node_ && node_->GetN() > 0) ? other) const { return !(*this == other); }
  VisitedNode_Iterator<is_const> begin() { return node_->GetM() : default_m;
   }
   uint32_t GetN() const *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }
  void operator++ { return node_ ? node_->GetN() : 0; }
   int GetNStarted() const {(); // Implementation in .cc
  Node* operator*(); // Implementation in .cc

 private:
   // Declaration return node_ ? node_->GetNStarted() : 0; }
   uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
   bool IsTerminal order fixed
   NodePtr parent_node_ = nullptr; // Store parent node pointer
   bool solid_ = false;() const { return node_ ? node_->IsTerminal() : false; }
   bool IsTbTerminal() const
   uint16_t total_count_ = 0;
   Node* node_ptr_ = { return node_ ? node_->IsTbTerminal() : false; }
   Node::Bounds GetBounds() const {
     return node_ ? node_->GetBounds()
                  : Node::Bounds{GameResult::BLACK_WON, nullptr; // Ptr to current node (or array start in solid)
   uint16_t current_idx_ = GameResult::WHITE_WON};
   }
   float GetP() const { return edge_ ? edge_->GetP() 0; // Index used primarily for solid mode
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return VisitedNode_Iterator<true>(this);
}
inline VisitedNode_ : 0.0f; }
   Move GetMove(bool flip = false) const {
     returnIterator<false> Node::VisitedNodes() {
  return VisitedNode_Iterator<false>(this);
 edge_ ? edge_->GetMove(flip) : Move();
   }
   // Returns U = numerator * p / (}

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  void MakeMove(Move move);
  void TrimTreeAtHead();
  bool ResetToPosition(const std::1 + N_started).
   // Passed numerator is expected to be equal to (cpuct * sqrt(N[string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToparent_started])).
   float GetU(float numerator) const {
       return GetP() == 0.0fPosition(const GameState& pos);
  const Position& HeadPosition() const { return history_.Last(); } ? 0.0f : numerator * GetP() / (1.0f + static_cast<float
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackTo>(GetNStarted()));
   }
   std::string DebugString() const;

 protected:
   Edge* edge_ = nullptr;
   Node* node_ = nullptr;
};


template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<Move() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  voidis_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  using NodePtr = std::conditional_t<is_const, const Node*, Node*>;
 DeallocateTree();
  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
};

}  // namespace classic
}  // namespace lczero
